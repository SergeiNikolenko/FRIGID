#!/usr/bin/env python
"""Train a staged DreaMS adapter as a serious MIST replacement candidate."""

import argparse
import csv
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dreams.api import PreTrainedModel
from dreams.definitions import DREAMS_EMBEDDING, SPECTRUM
import dreams.utils.data as du
import dreams.utils.dformats as dformats
from dreams.utils.data import MSData


try:
    from clearml import Logger, OutputModel, Task
except Exception:  # pragma: no cover - ClearML is optional for local parsing.
    Logger = None
    OutputModel = None
    Task = None


def parse_number_list(value: str, cast):
    if not value:
        return []
    return [cast(item) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a DreaMS encoder adapter with MIST anchoring, MIST-error "
            "focus, and full-validation fingerprint gates."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-spectra-hdf5", required=True)
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--train-targets-npz", required=True)
    parser.add_argument("--train-targets-key", default="ground_truth")
    parser.add_argument("--train-mist-npz", required=True)
    parser.add_argument("--train-mist-key", default="probs")
    parser.add_argument("--train-mist-metadata", required=True)
    parser.add_argument("--val-spectra-hdf5", required=True)
    parser.add_argument("--val-metadata", required=True)
    parser.add_argument("--val-targets-npz", required=True)
    parser.add_argument("--val-targets-key", default="ground_truth")
    parser.add_argument("--val-mist-npz", required=True)
    parser.add_argument("--val-mist-key", default="probs")
    parser.add_argument("--val-mist-metadata", required=True)
    parser.add_argument("--decoder-sensitive-csv", default=None)
    parser.add_argument("--decoder-sensitive-column", default="delta_tanimoto_top1_mist_binary_minus_ground_truth")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--adapter-hidden-dim", type=int, default=1024)
    parser.add_argument("--head-hidden-dim", type=int, default=2048)
    parser.add_argument("--fingerprint-bits", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--adapter-scale", type=float, default=0.5)

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--adapter-lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)

    parser.add_argument("--target-weight", type=float, default=1.0)
    parser.add_argument("--mist-anchor-weight", type=float, default=0.5)
    parser.add_argument("--mist-error-weight", type=float, default=4.0)
    parser.add_argument("--mist-uncertainty-weight", type=float, default=0.5)
    parser.add_argument("--false-negative-weight", type=float, default=2.0)
    parser.add_argument("--false-positive-weight", type=float, default=1.0)
    parser.add_argument("--decoder-sensitive-weight", type=float, default=2.0)
    parser.add_argument("--soft-tanimoto-weight", type=float, default=0.5)
    parser.add_argument("--count-loss-weight", type=float, default=0.02)
    parser.add_argument("--mist-threshold", type=float, default=0.25)
    parser.add_argument(
        "--encoder-trainable-regex",
        default=(
            r"(^head\.|backbone\.transformer_encoder\.(atts|ffs)\.[56]\."
            r"|backbone\.transformer_encoder\.scales\.(9|1[0-4])\.)"
        ),
        help="Regex selecting DreaMS encoder parameters to unfreeze after warmup.",
    )
    parser.add_argument(
        "--thresholds",
        default="0.02,0.03,0.04,0.05,0.075,0.1,0.125,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8",
    )
    parser.add_argument("--top-k", default="32,48,64,80,96,128,160,192,256,320")
    parser.add_argument("--clearml", action="store_true")
    parser.add_argument("--clearml-project", default="FRIGID/DreaMS Replacement")
    parser.add_argument("--clearml-task-name", default="FRIGID DreaMS adapter replacement MSG")
    return parser.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_npz_array(path: str, key: str) -> np.ndarray:
    arrays = np.load(path)
    if key not in arrays:
        raise ValueError(f"Array {key!r} not found in {path}; available: {arrays.files}")
    return arrays[key]


def load_metadata(path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    metadata = pd.read_csv(path)
    if "spec_name" not in metadata.columns:
        raise ValueError(f"Missing spec_name column in {path}")
    if max_rows is not None:
        metadata = metadata.iloc[:max_rows].copy()
    metadata["spec_name"] = metadata["spec_name"].astype(str)
    return metadata.reset_index(drop=True)


def align_array_by_spec_name(
    npz_path: str,
    key: str,
    metadata_path: str,
    target_spec_names: Iterable[str],
) -> np.ndarray:
    values = load_npz_array(npz_path, key).astype(np.float32)
    metadata = pd.read_csv(metadata_path)
    if "spec_name" not in metadata.columns:
        raise ValueError(f"Missing spec_name column in {metadata_path}")
    index = {str(spec_name): i for i, spec_name in enumerate(metadata["spec_name"])}
    target_spec_names = [str(spec_name) for spec_name in target_spec_names]
    missing = [spec_name for spec_name in target_spec_names if spec_name not in index]
    if missing:
        raise ValueError(f"{len(missing)} spec_names missing from {metadata_path}: {missing[:5]}")
    return values[[index[spec_name] for spec_name in target_spec_names]]


def load_targets(npz_path: str, key: str, max_rows: Optional[int] = None) -> np.ndarray:
    targets = load_npz_array(npz_path, key).astype(np.float32)
    if max_rows is not None:
        targets = targets[:max_rows]
    return targets


def load_decoder_case_weights(
    csv_path: Optional[str],
    spec_names: List[str],
    column: str,
    scale: float,
) -> np.ndarray:
    weights = np.ones(len(spec_names), dtype=np.float32)
    if not csv_path:
        return weights
    table = pd.read_csv(csv_path)
    if "spec_name" not in table.columns or column not in table.columns:
        raise ValueError(f"{csv_path} must contain spec_name and {column}")
    values_by_name = {
        str(row["spec_name"]): float(row[column])
        for _, row in table[["spec_name", column]].dropna().iterrows()
    }
    raw = np.array([values_by_name.get(spec_name, 0.0) for spec_name in spec_names], dtype=np.float32)
    difficulty = np.maximum(-raw, 0.0)
    if difficulty.max() > 0:
        difficulty = difficulty / difficulty.max()
    return weights + scale * difficulty


def make_bit_weights(
    mist_probs: np.ndarray,
    targets: np.ndarray,
    case_weights: np.ndarray,
    args,
) -> np.ndarray:
    mist_binary = (mist_probs >= args.mist_threshold).astype(np.float32)
    target_binary = (targets > 0.5).astype(np.float32)
    false_negative = (mist_binary == 0) & (target_binary == 1)
    false_positive = (mist_binary == 1) & (target_binary == 0)
    uncertainty = 1.0 - np.minimum(np.abs(mist_probs - 0.5) * 2.0, 1.0)

    weights = np.ones_like(targets, dtype=np.float32)
    weights += args.mist_error_weight * (false_negative | false_positive).astype(np.float32)
    weights += args.false_negative_weight * false_negative.astype(np.float32)
    weights += args.false_positive_weight * false_positive.astype(np.float32)
    weights += args.mist_uncertainty_weight * uncertainty.astype(np.float32)
    weights *= case_weights[:, None].astype(np.float32)
    return weights.astype(np.float32)


def metrics_from_counts(intersection, pred_counts, target_counts, fingerprint_bits):
    union = pred_counts + target_counts - intersection
    tanimoto = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )
    tp = float(intersection.sum())
    predicted_total = float(pred_counts.sum())
    target_total = float(target_counts.sum())
    fp = predicted_total - tp
    fn = target_total - tp
    total_bits = float(len(target_counts) * fingerprint_bits)
    precision = tp / predicted_total if predicted_total > 0 else 0.0
    recall = tp / target_total if target_total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "mean_tanimoto": float(tanimoto.mean()) if len(tanimoto) else 0.0,
        "median_tanimoto": float(np.median(tanimoto)) if len(tanimoto) else 0.0,
        "bit_accuracy": float(1.0 - ((fp + fn) / total_bits)) if total_bits else 0.0,
        "bit_precision": precision,
        "bit_recall": recall,
        "bit_f1": f1,
        "mean_active_bits": float(pred_counts.mean()) if len(pred_counts) else 0.0,
        "mean_target_active_bits": float(target_counts.mean()) if len(target_counts) else 0.0,
    }


def threshold_metrics(probs, targets, target_counts, threshold):
    pred = probs >= threshold
    intersection = np.logical_and(pred, targets).sum(axis=1)
    pred_counts = pred.sum(axis=1)
    return metrics_from_counts(intersection, pred_counts, target_counts, targets.shape[1])


def topk_metrics(probs, targets, target_counts, top_ks):
    if not top_ks:
        return {}
    max_k = min(max(int(k) for k in top_ks), probs.shape[1])
    top_indices = np.argpartition(probs, -max_k, axis=1)[:, -max_k:]
    top_scores = np.take_along_axis(probs, top_indices, axis=1)
    order = np.argsort(top_scores, axis=1)[:, ::-1]
    top_indices = np.take_along_axis(top_indices, order, axis=1)
    row_indices = np.arange(probs.shape[0])[:, None]
    hit_cumsum = targets[row_indices, top_indices].cumsum(axis=1)
    rows = {}
    for k in top_ks:
        k = min(max(int(k), 1), probs.shape[1])
        intersection = hit_cumsum[:, k - 1]
        pred_counts = np.full(len(targets), k, dtype=np.int64)
        rows[k] = metrics_from_counts(intersection, pred_counts, target_counts, targets.shape[1])
    return rows


def sweep_probs(probs, targets, thresholds, top_ks):
    target_binary = targets > 0.5
    target_counts = target_binary.sum(axis=1)
    rows = []
    for threshold in thresholds:
        rows.append({"mode": "threshold", "value": threshold, **threshold_metrics(probs, target_binary, target_counts, threshold)})
    topk_rows = topk_metrics(probs, target_binary, target_counts, top_ks)
    for k in top_ks:
        rows.append({"mode": "top_k", "value": k, **topk_rows[k]})
    return sorted(rows, key=lambda row: row["mean_tanimoto"], reverse=True)


def write_rows_csv(path: Path, rows: List[Dict]):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


class DreaMSAdapterDataset(Dataset):
    def __init__(
        self,
        spectra_hdf5: str,
        metadata: pd.DataFrame,
        targets: np.ndarray,
        mist_probs: np.ndarray,
        weights: np.ndarray,
    ):
        if not (len(metadata) == len(targets) == len(mist_probs) == len(weights)):
            raise ValueError(
                "Dataset length mismatch: "
                f"metadata={len(metadata)}, targets={len(targets)}, "
                f"mist={len(mist_probs)}, weights={len(weights)}"
            )
        spec_preproc = du.SpectrumPreprocessor(
            dformat=dformats.DataFormatA(),
            n_highest_peaks=100,
        )
        ms_data = MSData(Path(spectra_hdf5), mode="r", in_mem=True)
        self.spectra = ms_data.to_torch_dataset(spec_preproc)
        if len(self.spectra) < len(metadata):
            raise ValueError(f"Spectra rows ({len(self.spectra)}) are fewer than metadata rows ({len(metadata)})")
        self.metadata = metadata
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.mist_probs = torch.as_tensor(mist_probs, dtype=torch.float32)
        self.weights = torch.as_tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        return (
            self.spectra[idx][SPECTRUM],
            self.targets[idx],
            self.mist_probs[idx],
            self.weights[idx],
        )


class ResidualAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, scale: float):
        super().__init__()
        self.scale = float(scale)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, embeddings):
        return embeddings + self.scale * self.net(embeddings)


class DreaMSAdapterFingerprintModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.spec_encoder = PreTrainedModel.from_name(DREAMS_EMBEDDING).model
        self.adapter = ResidualAdapter(
            input_dim=args.input_dim,
            hidden_dim=args.adapter_hidden_dim,
            dropout=args.dropout,
            scale=args.adapter_scale,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(args.input_dim),
            nn.Linear(args.input_dim, args.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.head_hidden_dim, args.fingerprint_bits),
        )

    def forward(self, spectra):
        embeddings = self.spec_encoder(spectra)
        adapted = self.adapter(embeddings)
        return self.head(adapted)


def set_encoder_trainability(model: DreaMSAdapterFingerprintModel, epoch: int, args) -> Dict[str, int]:
    trainable_regex = re.compile(args.encoder_trainable_regex)
    unfreeze = epoch > args.warmup_epochs
    matched = 0
    total = 0
    for name, param in model.spec_encoder.named_parameters():
        total += 1
        should_train = bool(unfreeze and trainable_regex.search(name))
        param.requires_grad = should_train
        if should_train:
            matched += 1
    if unfreeze:
        model.spec_encoder.train()
    else:
        model.spec_encoder.eval()
    return {"encoder_param_tensors": total, "trainable_encoder_param_tensors": matched}


def soft_tanimoto_loss(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=1)
    union = (probs + targets - probs * targets).sum(dim=1)
    return 1.0 - ((intersection + eps) / (union + eps)).mean()


def active_count_loss(logits, targets):
    probs = torch.sigmoid(logits)
    pred_count = probs.sum(dim=1)
    target_count = targets.sum(dim=1).clamp_min(1.0)
    return torch.square((pred_count - target_count) / target_count).mean()


def encoder_has_trainable_parameters(model: DreaMSAdapterFingerprintModel) -> bool:
    return any(param.requires_grad for param in model.spec_encoder.parameters())


def train_epoch(model, loader, optimizer, device, args):
    model.train()
    if not encoder_has_trainable_parameters(model):
        model.spec_encoder.eval()
    bce_none = nn.BCEWithLogitsLoss(reduction="none")
    losses = []
    parts = {"target": [], "mist_anchor": [], "soft_tanimoto": [], "count": []}
    progress = tqdm(loader, desc="train", leave=False)
    for spectra, targets, mist_probs, weights in progress:
        spectra = spectra.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mist_probs = mist_probs.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)

        logits = model(spectra)
        target_loss = (bce_none(logits, targets) * weights).mean()
        mist_anchor = nn.functional.binary_cross_entropy_with_logits(logits, mist_probs)
        tan_loss = soft_tanimoto_loss(logits, targets)
        count_loss = active_count_loss(logits, targets)
        loss = (
            args.target_weight * target_loss
            + args.mist_anchor_weight * mist_anchor
            + args.soft_tanimoto_weight * tan_loss
            + args.count_loss_weight * count_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()

        losses.append(float(loss.item()))
        parts["target"].append(float(target_loss.item()))
        parts["mist_anchor"].append(float(mist_anchor.item()))
        parts["soft_tanimoto"].append(float(tan_loss.item()))
        parts["count"].append(float(count_loss.item()))
        progress.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}")

    return {
        "train_loss": float(np.mean(losses)),
        "train_target_loss": float(np.mean(parts["target"])),
        "train_mist_anchor_loss": float(np.mean(parts["mist_anchor"])),
        "train_soft_tanimoto_loss": float(np.mean(parts["soft_tanimoto"])),
        "train_count_loss": float(np.mean(parts["count"])),
    }


def evaluate(model, loader, device, targets_np, thresholds, top_ks):
    model.eval()
    probs = []
    with torch.no_grad():
        for spectra, _targets, _mist_probs, _weights in tqdm(loader, desc="val", leave=False):
            logits = model(spectra.to(device, non_blocking=True))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    probs_np = np.vstack(probs).astype(np.float32)
    return sweep_probs(probs_np, targets_np, thresholds, top_ks), probs_np


def build_task(args):
    if not args.clearml:
        return None, None
    if Task is None:
        raise RuntimeError("ClearML was requested but clearml is not importable")
    task = Task.init(
        project_name=args.clearml_project,
        task_name=args.clearml_task_name,
        reuse_last_task_id=False,
    )
    task.add_tags(["frigid", "dreams-adapter", "mist-anchor", "decoder-sensitive", "msg-full"])
    task.connect(vars(args))
    return task, Logger.current_logger()


def log_scalar(logger, title, series, iteration, value):
    if logger is not None:
        logger.report_scalar(title, series, iteration=iteration, value=value)


def save_checkpoint(path: Path, model, optimizer, epoch, args, record, best, trainability):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "record": record,
            "best": best,
            "trainability": trainability,
        },
        path,
    )


def main():
    args = parse_args()
    seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds = parse_number_list(args.thresholds, float)
    top_ks = parse_number_list(args.top_k, int)
    task, logger = build_task(args)

    train_metadata = load_metadata(args.train_metadata, args.max_train_rows)
    val_metadata = load_metadata(args.val_metadata, args.max_val_rows)
    train_spec_names = train_metadata["spec_name"].tolist()
    val_spec_names = val_metadata["spec_name"].tolist()

    train_targets = load_targets(args.train_targets_npz, args.train_targets_key, args.max_train_rows)
    val_targets = load_targets(args.val_targets_npz, args.val_targets_key, args.max_val_rows)
    train_mist = align_array_by_spec_name(args.train_mist_npz, args.train_mist_key, args.train_mist_metadata, train_spec_names)
    val_mist = align_array_by_spec_name(args.val_mist_npz, args.val_mist_key, args.val_mist_metadata, val_spec_names)
    case_weights = load_decoder_case_weights(
        args.decoder_sensitive_csv,
        train_spec_names,
        args.decoder_sensitive_column,
        args.decoder_sensitive_weight,
    )
    train_weights = make_bit_weights(train_mist, train_targets, case_weights, args)
    val_weights = np.ones_like(val_targets, dtype=np.float32)

    mist_baseline_rows = sweep_probs(val_mist, val_targets, thresholds, top_ks)
    (output_dir / "mist_baseline_metrics.json").write_text(json.dumps({"rows": mist_baseline_rows}, indent=2))
    write_rows_csv(output_dir / "mist_baseline_metrics.csv", mist_baseline_rows)
    print("Best MIST baseline:", json.dumps(mist_baseline_rows[0], indent=2), flush=True)

    train_dataset = DreaMSAdapterDataset(
        args.train_spectra_hdf5,
        train_metadata,
        train_targets,
        train_mist,
        train_weights,
    )
    val_dataset = DreaMSAdapterDataset(
        args.val_spectra_hdf5,
        val_metadata,
        val_targets,
        val_mist,
        val_weights,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DreaMSAdapterFingerprintModel(args).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.spec_encoder.parameters(), "lr": args.encoder_lr, "name": "encoder"},
            {"params": model.adapter.parameters(), "lr": args.adapter_lr, "name": "adapter"},
            {"params": model.head.parameters(), "lr": args.head_lr, "name": "head"},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best = {
        "epoch": 0,
        "best_metrics": {"mean_tanimoto": -1.0},
        "best_delta_vs_mist": -1.0,
    }
    history = []
    epochs_without_improvement = 0

    summary_static = {
        "train_rows": int(len(train_metadata)),
        "val_rows": int(len(val_metadata)),
        "fingerprint_bits": int(train_targets.shape[1]),
        "mist_baseline_best": mist_baseline_rows[0],
        "device": str(device),
        "args": vars(args),
    }
    (output_dir / "run_config.json").write_text(json.dumps(summary_static, indent=2))

    for epoch in range(1, args.epochs + 1):
        trainability = set_encoder_trainability(model, epoch, args)
        print(f"Epoch {epoch} trainability: {trainability}", flush=True)
        train_record = train_epoch(model, train_loader, optimizer, device, args)
        rows, val_probs = evaluate(model, val_loader, device, val_targets, thresholds, top_ks)
        epoch_best = rows[0]
        delta_vs_mist = float(epoch_best["mean_tanimoto"] - mist_baseline_rows[0]["mean_tanimoto"])
        scheduler.step(epoch_best["mean_tanimoto"])

        record = {
            "epoch": epoch,
            **train_record,
            "best_val": epoch_best,
            "delta_vs_mist": delta_vs_mist,
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            "trainability": trainability,
        }
        history.append(record)
        print(json.dumps(record, indent=2), flush=True)
        for key, value in train_record.items():
            log_scalar(logger, "Train", key, epoch, value)
        log_scalar(logger, "Validation", "best_mean_tanimoto", epoch, epoch_best["mean_tanimoto"])
        log_scalar(logger, "Validation", "delta_vs_mist", epoch, delta_vs_mist)

        save_checkpoint(output_dir / "last_model.pt", model, optimizer, epoch, args, record, best, trainability)
        if epoch_best["mean_tanimoto"] > best["best_metrics"]["mean_tanimoto"]:
            best = {
                "epoch": epoch,
                "best_metrics": epoch_best,
                "best_delta_vs_mist": delta_vs_mist,
            }
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "best_model.pt", model, optimizer, epoch, args, record, best, trainability)
            np.savez_compressed(
                output_dir / "val_predictions.npz",
                probs=val_probs,
                dreams_adapter_probs=val_probs,
            )
            val_metadata.to_csv(output_dir / "val_prediction_metadata.csv", index=False)
            (output_dir / "val_sweep_metrics.json").write_text(json.dumps({"rows": rows}, indent=2))
            write_rows_csv(output_dir / "val_sweep_metrics.csv", rows)
        else:
            epochs_without_improvement += 1

        summary = {
            **summary_static,
            "best": best,
            "history": history,
            "passed_mist_replacement_gate": bool(best["best_delta_vs_mist"] >= 0.005),
            "gate_delta": 0.005,
        }
        (output_dir / "training_metrics.json").write_text(json.dumps(summary, indent=2))

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epochs_without_improvement} epochs without improvement", flush=True)
            break

    final_summary = {
        **summary_static,
        "best": best,
        "history": history,
        "passed_mist_replacement_gate": bool(best["best_delta_vs_mist"] >= 0.005),
        "gate_delta": 0.005,
    }
    (output_dir / "summary.json").write_text(json.dumps(final_summary, indent=2))
    print(json.dumps(final_summary["best"], indent=2), flush=True)

    if task is not None and OutputModel is not None:
        output_model = OutputModel(task=task, name="FRIGID DreaMS adapter best model")
        output_model.update_weights(weights_filename=str(output_dir / "best_model.pt"), auto_delete_file=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
