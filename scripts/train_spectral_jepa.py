#!/usr/bin/env python
"""Train a JEPA-style spectrum encoder and evaluate it with a fingerprint head."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dreams.definitions import SPECTRUM
import dreams.utils.data as du
import dreams.utils.dformats as dformats
from dreams.utils.data import MSData

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from frigid.dreams_fingerprint_head import (  # noqa: E402
    FingerprintHead,
    compute_fingerprint_metrics,
)


def parse_number_list(value: str, cast):
    if not value:
        return []
    return [cast(item) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "JEPA-like pretraining for MS/MS spectra followed by a frozen-encoder "
            "fingerprint-head gate."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-spectra-hdf5", required=True)
    parser.add_argument("--train-metadata", required=True)
    parser.add_argument("--train-targets-npz", required=True)
    parser.add_argument("--train-targets-key", default="ground_truth")
    parser.add_argument("--val-spectra-hdf5", required=True)
    parser.add_argument("--val-metadata", required=True)
    parser.add_argument("--val-targets-npz", required=True)
    parser.add_argument("--val-targets-key", default="ground_truth")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-val-rows", type=int, default=None)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--input-features", type=int, default=2)
    parser.add_argument("--mz-scale", type=float, default=1000.0)

    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--transformer-layers", type=int, default=6)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--predictor-hidden-dim", type=int, default=768)

    parser.add_argument("--pretrain-epochs", type=int, default=50)
    parser.add_argument("--head-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pretrain-lr", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--target-fraction", type=float, default=0.25)
    parser.add_argument("--target-window-fraction", type=float, default=0.18)
    parser.add_argument("--mask-modes", default="mz_window,high_intensity,random")
    parser.add_argument("--latent-loss", choices=["mse", "smooth_l1", "cosine"], default="smooth_l1")
    parser.add_argument("--variance-weight", type=float, default=0.05)
    parser.add_argument("--covariance-weight", type=float, default=0.005)
    parser.add_argument("--variance-eps", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--head-hidden-dim", type=int, default=1024)
    parser.add_argument("--head-dropout", type=float, default=0.1)
    parser.add_argument("--fingerprint-threshold", type=float, default=0.5)
    parser.add_argument("--disable-pos-weight", action="store_true")
    parser.add_argument("--thresholds", default="0.02,0.03,0.04,0.05,0.075,0.1,0.125,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8")
    parser.add_argument("--top-k", default="32,48,64,80,96,128,160,192,256,320")
    return parser.parse_args()


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_npz_array(path: str, key: str, max_rows: Optional[int] = None) -> np.ndarray:
    arrays = np.load(path)
    if key not in arrays:
        raise ValueError(f"Array {key!r} not found in {path}; available: {arrays.files}")
    values = arrays[key].astype(np.float32)
    if max_rows is not None:
        values = values[:max_rows]
    return values


def load_metadata(path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    metadata = pd.read_csv(path)
    if max_rows is not None:
        metadata = metadata.iloc[:max_rows].copy()
    return metadata.reset_index(drop=True)


class SpectrumFingerprintDataset(Dataset):
    def __init__(
        self,
        spectra_hdf5: str,
        metadata_path: str,
        targets_npz: str,
        targets_key: str,
        n_highest_peaks: int,
        max_rows: Optional[int] = None,
    ):
        self.metadata = load_metadata(metadata_path, max_rows)
        targets = load_npz_array(targets_npz, targets_key, max_rows)
        if len(self.metadata) != len(targets):
            raise ValueError(
                f"Metadata rows ({len(self.metadata)}) do not match target rows ({len(targets)})"
            )
        spec_preproc = du.SpectrumPreprocessor(
            dformat=dformats.DataFormatA(),
            n_highest_peaks=n_highest_peaks,
        )
        ms_data = MSData(Path(spectra_hdf5), mode="r", in_mem=True)
        spectra = ms_data.to_torch_dataset(spec_preproc)
        if len(spectra) < len(self.metadata):
            raise ValueError(f"Spectra rows ({len(spectra)}) are fewer than metadata rows ({len(self.metadata)})")
        self.spectra = spectra
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self.spectra[index][SPECTRUM], self.targets[index], index


def coerce_peak_batch(spectra: torch.Tensor, input_features: int) -> torch.Tensor:
    if spectra.ndim != 3:
        raise ValueError(f"Expected batched spectra to be 3D, got {tuple(spectra.shape)}")
    if spectra.shape[-1] >= input_features:
        peaks = spectra[..., :input_features]
    elif spectra.shape[1] >= input_features:
        peaks = spectra.transpose(1, 2)[..., :input_features]
    else:
        raise ValueError(f"Cannot find {input_features} input features in spectra shape {tuple(spectra.shape)}")
    return peaks.float()


def normalize_peaks(peaks: torch.Tensor, mz_scale: float) -> torch.Tensor:
    peaks = peaks.clone()
    peaks[..., 0] = peaks[..., 0] / mz_scale
    if peaks.shape[-1] > 1:
        peaks[..., 1] = torch.log1p(torch.clamp(peaks[..., 1], min=0.0))
    return peaks


def valid_peak_mask(peaks: torch.Tensor) -> torch.Tensor:
    if peaks.shape[-1] > 1:
        return peaks[..., 1] > 0
    return peaks[..., 0] > 0


class SpectrumSetEncoder(nn.Module):
    def __init__(
        self,
        input_features: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        mz_scale: float,
    ):
        super().__init__()
        self.input_features = input_features
        self.mz_scale = mz_scale
        self.token_projection = nn.Sequential(
            nn.LayerNorm(input_features),
            nn.Linear(input_features, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, spectra: torch.Tensor, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        peaks = coerce_peak_batch(spectra, self.input_features)
        base_valid = valid_peak_mask(peaks)
        if token_mask is None:
            token_mask = base_valid
        else:
            token_mask = token_mask & base_valid
        features = normalize_peaks(peaks, self.mz_scale)
        features = features * token_mask.unsqueeze(-1).to(features.dtype)
        tokens = self.token_projection(features)
        tokens = self.transformer(tokens, src_key_padding_mask=~token_mask)
        denom = token_mask.sum(dim=1, keepdim=True).clamp_min(1).to(tokens.dtype)
        pooled = (tokens * token_mask.unsqueeze(-1).to(tokens.dtype)).sum(dim=1) / denom
        return self.output_norm(pooled)


class JEPAPredictor(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, context_repr: torch.Tensor) -> torch.Tensor:
        return self.net(context_repr)


def choose_masks(peaks: torch.Tensor, modes: List[str], target_fraction: float, window_fraction: float):
    valid = valid_peak_mask(peaks)
    batch, n_peaks = valid.shape
    target = torch.zeros_like(valid)
    mz = peaks[..., 0]
    intensity = peaks[..., 1] if peaks.shape[-1] > 1 else torch.ones_like(mz)

    for row in range(batch):
        valid_idx = torch.nonzero(valid[row], as_tuple=False).flatten()
        if len(valid_idx) == 0:
            continue
        mode = random.choice(modes)
        count = max(1, int(math.ceil(len(valid_idx) * target_fraction)))
        if mode == "high_intensity":
            ranked = valid_idx[torch.argsort(intensity[row, valid_idx], descending=True)]
            selected = ranked[:count]
        elif mode == "mz_window":
            row_mz = mz[row, valid_idx]
            span = (row_mz.max() - row_mz.min()).clamp_min(1.0)
            width = span * window_fraction
            center = row_mz[random.randrange(len(row_mz))]
            in_window = valid_idx[(row_mz >= center - width / 2) & (row_mz <= center + width / 2)]
            if len(in_window) == 0:
                selected = valid_idx[torch.randperm(len(valid_idx), device=valid_idx.device)[:count]]
            else:
                selected = in_window
        else:
            selected = valid_idx[torch.randperm(len(valid_idx), device=valid_idx.device)[:count]]
        target[row, selected] = True
        if bool((valid[row] & ~target[row]).sum() == 0):
            target[row, selected[-1]] = False

    context = valid & ~target
    empty_context = context.sum(dim=1) == 0
    if empty_context.any():
        context[empty_context] = valid[empty_context]
    empty_target = target.sum(dim=1) == 0
    if empty_target.any():
        target[empty_target] = valid[empty_target]
    return context, target


def latent_prediction_loss(predicted: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "mse":
        return F.mse_loss(predicted, target)
    if kind == "cosine":
        return 1.0 - F.cosine_similarity(predicted, target, dim=1).mean()
    return F.smooth_l1_loss(predicted, target)


def variance_loss(values: torch.Tensor, eps: float) -> torch.Tensor:
    if values.shape[0] < 2:
        return values.new_tensor(0.0)
    std = torch.sqrt(values.var(dim=0, unbiased=False) + eps)
    return torch.mean(F.relu(1.0 - std))


def covariance_loss(values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] < 2:
        return values.new_tensor(0.0)
    values = values - values.mean(dim=0)
    cov = (values.T @ values) / (values.shape[0] - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / values.shape[1]


@torch.no_grad()
def update_ema(source: nn.Module, target: nn.Module, decay: float):
    for src_param, tgt_param in zip(source.parameters(), target.parameters()):
        tgt_param.data.mul_(decay).add_(src_param.data, alpha=1.0 - decay)


def train_jepa(args, train_loader: DataLoader, device: torch.device, output_dir: Path):
    modes = [mode.strip() for mode in args.mask_modes.split(",") if mode.strip()]
    context_encoder = SpectrumSetEncoder(
        input_features=args.input_features,
        embed_dim=args.embed_dim,
        num_layers=args.transformer_layers,
        num_heads=args.attention_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        mz_scale=args.mz_scale,
    ).to(device)
    target_encoder = deepcopy(context_encoder).to(device)
    for param in target_encoder.parameters():
        param.requires_grad = False
    predictor = JEPAPredictor(args.embed_dim, args.predictor_hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        list(context_encoder.parameters()) + list(predictor.parameters()),
        lr=args.pretrain_lr,
        weight_decay=args.weight_decay,
    )

    history = []
    best_loss = float("inf")
    best_path = output_dir / "best_jepa_encoder.pt"
    for epoch in range(1, args.pretrain_epochs + 1):
        context_encoder.train()
        predictor.train()
        target_encoder.eval()
        epoch_rows = []
        for spectra, _, _ in tqdm(train_loader, desc=f"JEPA epoch {epoch}/{args.pretrain_epochs}"):
            spectra = spectra.to(device)
            peaks = coerce_peak_batch(spectra, args.input_features)
            context_mask, target_mask = choose_masks(peaks, modes, args.target_fraction, args.target_window_fraction)
            context_mask = context_mask.to(device)
            target_mask = target_mask.to(device)

            context_repr = context_encoder(spectra, context_mask)
            with torch.no_grad():
                target_repr = target_encoder(spectra, target_mask)
            predicted_repr = predictor(context_repr)

            latent_loss = latent_prediction_loss(predicted_repr, target_repr, args.latent_loss)
            var_loss = variance_loss(context_repr, args.variance_eps) + variance_loss(predicted_repr, args.variance_eps)
            cov_loss = covariance_loss(context_repr) + covariance_loss(predicted_repr)
            loss = latent_loss + args.variance_weight * var_loss + args.covariance_weight * cov_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.gradient_clip:
                torch.nn.utils.clip_grad_norm_(list(context_encoder.parameters()) + list(predictor.parameters()), args.gradient_clip)
            optimizer.step()
            update_ema(context_encoder, target_encoder, args.ema_decay)

            epoch_rows.append(
                {
                    "loss": float(loss.item()),
                    "latent_loss": float(latent_loss.item()),
                    "variance_loss": float(var_loss.item()),
                    "covariance_loss": float(cov_loss.item()),
                }
            )

        row = {
            "epoch": epoch,
            "loss": float(np.mean([item["loss"] for item in epoch_rows])),
            "latent_loss": float(np.mean([item["latent_loss"] for item in epoch_rows])),
            "variance_loss": float(np.mean([item["variance_loss"] for item in epoch_rows])),
            "covariance_loss": float(np.mean([item["covariance_loss"] for item in epoch_rows])),
        }
        history.append(row)
        if row["loss"] < best_loss:
            best_loss = row["loss"]
            torch.save(
                {
                    "state_dict": context_encoder.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "metrics": row,
                },
                best_path,
            )

    torch.save(
        {
            "state_dict": context_encoder.state_dict(),
            "target_state_dict": target_encoder.state_dict(),
            "predictor_state_dict": predictor.state_dict(),
            "args": vars(args),
            "history": history,
        },
        output_dir / "last_jepa_encoder.pt",
    )
    (output_dir / "pretrain_metrics.json").write_text(json.dumps({"history": history, "best_checkpoint": str(best_path)}, indent=2))
    checkpoint = torch.load(best_path, map_location=device)
    context_encoder.load_state_dict(checkpoint["state_dict"])
    return context_encoder, history


def topk_predictions(probs: np.ndarray, k: int) -> np.ndarray:
    pred = np.zeros_like(probs, dtype=np.float32)
    if k <= 0:
        return pred
    top = np.argpartition(-probs, kth=min(k, probs.shape[1] - 1), axis=1)[:, :k]
    rows = np.arange(probs.shape[0])[:, None]
    pred[rows, top] = 1.0
    return pred


def sweep_predictions(probs: np.ndarray, targets: np.ndarray, thresholds: List[float], top_ks: List[int]) -> List[Dict]:
    rows = []
    for threshold in thresholds:
        metrics, _ = compute_fingerprint_metrics(probs, targets, threshold)
        rows.append({"mode": "threshold", "value": threshold, **metrics.__dict__})
    for k in top_ks:
        pred = topk_predictions(probs, k)
        metrics, _ = compute_fingerprint_metrics(pred, targets, 0.5)
        rows.append({"mode": "top_k", "value": k, **metrics.__dict__})
    return sorted(rows, key=lambda item: item["mean_tanimoto"], reverse=True)


def write_rows_csv(path: Path, rows: List[Dict]):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def encode_dataset(encoder: SpectrumSetEncoder, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    encoder.eval()
    embeddings = []
    targets = []
    indices = []
    for spectra, batch_targets, batch_indices in tqdm(loader, desc="Encoding spectra"):
        spectra = spectra.to(device)
        embeddings.append(encoder(spectra).cpu().numpy())
        targets.append(batch_targets.numpy())
        indices.append(batch_indices.numpy())
    return np.vstack(embeddings), np.vstack(targets), np.concatenate(indices)


def evaluate_head(head: FingerprintHead, embeddings: np.ndarray, targets: np.ndarray, batch_size: int, device: torch.device, threshold: float):
    head.eval()
    probs = []
    losses = []
    criterion = torch.nn.BCEWithLogitsLoss()
    for start in range(0, len(embeddings), batch_size):
        batch_embeddings = torch.as_tensor(embeddings[start : start + batch_size], dtype=torch.float32, device=device)
        batch_targets = torch.as_tensor(targets[start : start + batch_size], dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = head(batch_embeddings)
            losses.append(float(criterion(logits, batch_targets).item()))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    probs_np = np.vstack(probs)
    metrics, _ = compute_fingerprint_metrics(probs_np, targets, threshold)
    return float(np.mean(losses)), metrics, probs_np


def train_fingerprint_head(args, encoder: SpectrumSetEncoder, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, output_dir: Path):
    for param in encoder.parameters():
        param.requires_grad = False
    train_embeddings, train_targets, _ = encode_dataset(encoder, train_loader, device)
    val_embeddings, val_targets, val_indices = encode_dataset(encoder, val_loader, device)

    head = FingerprintHead(
        input_dim=train_embeddings.shape[1],
        fingerprint_bits=train_targets.shape[1],
        hidden_dim=args.head_hidden_dim,
        dropout=args.head_dropout,
    ).to(device)
    if args.disable_pos_weight:
        pos_weight_value = 1.0
        pos_weight = None
    else:
        positives = float(train_targets.sum())
        negatives = float(train_targets.size - positives)
        pos_weight_value = negatives / positives if positives > 0 else 1.0
        pos_weight = torch.full((train_targets.shape[1],), pos_weight_value, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)

    history = []
    best_metric = -1.0
    best_path = output_dir / "best_fingerprint_head.pt"
    for epoch in range(1, args.head_epochs + 1):
        head.train()
        order = np.random.permutation(len(train_embeddings))
        losses = []
        for start in tqdm(range(0, len(order), args.batch_size), desc=f"Head epoch {epoch}/{args.head_epochs}"):
            idx = order[start : start + args.batch_size]
            batch_embeddings = torch.as_tensor(train_embeddings[idx], dtype=torch.float32, device=device)
            batch_targets = torch.as_tensor(train_targets[idx], dtype=torch.float32, device=device)
            logits = head(batch_embeddings)
            loss = criterion(logits, batch_targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.gradient_clip:
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.gradient_clip)
            optimizer.step()
            losses.append(float(loss.item()))

        val_loss, val_metrics, val_probs = evaluate_head(
            head, val_embeddings, val_targets, args.batch_size, device, args.fingerprint_threshold
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            **val_metrics.__dict__,
        }
        history.append(row)
        if val_metrics.mean_tanimoto > best_metric:
            best_metric = val_metrics.mean_tanimoto
            torch.save(
                {
                    "state_dict": head.state_dict(),
                    "input_dim": train_embeddings.shape[1],
                    "fingerprint_bits": train_targets.shape[1],
                    "hidden_dim": args.head_hidden_dim,
                    "dropout": args.head_dropout,
                    "threshold": args.fingerprint_threshold,
                    "metrics": row,
                },
                best_path,
            )
            np.savez_compressed(output_dir / "val_predictions.npz", probs=val_probs, ground_truth=val_targets)

    checkpoint = torch.load(best_path, map_location=device)
    head.load_state_dict(checkpoint["state_dict"])
    _, _, val_probs = evaluate_head(head, val_embeddings, val_targets, args.batch_size, device, args.fingerprint_threshold)
    sweep = sweep_predictions(
        val_probs,
        val_targets,
        parse_number_list(args.thresholds, float),
        parse_number_list(args.top_k, int),
    )
    (output_dir / "val_sweep_metrics.json").write_text(json.dumps(sweep, indent=2))
    write_rows_csv(output_dir / "val_sweep_metrics.csv", sweep)
    (output_dir / "head_metrics.json").write_text(
        json.dumps(
            {
                "train_rows": int(len(train_embeddings)),
                "val_rows": int(len(val_embeddings)),
                "embedding_dim": int(train_embeddings.shape[1]),
                "fingerprint_bits": int(train_targets.shape[1]),
                "pos_weight": pos_weight_value,
                "history": history,
                "best_checkpoint": str(best_path),
                "best_sweep_row": sweep[0],
            },
            indent=2,
        )
    )
    return history, sweep, val_indices


def write_val_metadata(path: Path, val_metadata: pd.DataFrame, val_indices: np.ndarray):
    val_metadata.iloc[val_indices].reset_index(drop=True).to_csv(path, index=False)


def main():
    args = parse_args()
    seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = SpectrumFingerprintDataset(
        args.train_spectra_hdf5,
        args.train_metadata,
        args.train_targets_npz,
        args.train_targets_key,
        args.n_highest_peaks,
        args.max_train_rows,
    )
    val_dataset = SpectrumFingerprintDataset(
        args.val_spectra_hdf5,
        args.val_metadata,
        args.val_targets_npz,
        args.val_targets_key,
        args.n_highest_peaks,
        args.max_val_rows,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_config = {
        "args": vars(args),
        "device": str(device),
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    encoder, pretrain_history = train_jepa(args, train_loader, device, output_dir)
    head_history, sweep, val_indices = train_fingerprint_head(
        args, encoder, train_eval_loader, val_loader, device, output_dir
    )
    write_val_metadata(output_dir / "val_prediction_metadata.csv", val_dataset.metadata, val_indices)
    summary = {
        "train_rows": len(train_dataset),
        "val_rows": len(val_dataset),
        "pretrain_final": pretrain_history[-1] if pretrain_history else None,
        "head_final": head_history[-1] if head_history else None,
        "best_sweep_row": sweep[0],
        "promotion_note": "Compare best_sweep_row.mean_tanimoto against MIST baseline before any DLM benchmark.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
