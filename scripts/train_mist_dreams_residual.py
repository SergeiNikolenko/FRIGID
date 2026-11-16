#!/usr/bin/env python
"""Train a residual DreaMS adapter on top of MIST fingerprint logits."""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
scripts_path = os.path.join(project_root, "scripts")
for path in (src_path, scripts_path):
    if path not in sys.path:
        sys.path.insert(0, path)

from frigid.dreams_fingerprint_head import load_npz_array  # noqa: E402
from blend_fingerprint_predictions import (  # noqa: E402
    parse_number_list,
    threshold_metrics,
    topk_metrics,
)


class MistDreamsResidualAdapter(nn.Module):
    """Predict a bounded residual correction to MIST fingerprint logits."""

    def __init__(
        self,
        embedding_dim,
        fingerprint_bits,
        hidden_dim,
        dropout,
        residual_scale,
    ):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.embedding_net = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.mist_net = nn.Sequential(
            nn.LayerNorm(fingerprint_bits),
            nn.Linear(fingerprint_bits, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fingerprint_bits),
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, embeddings, mist_logits):
        features = torch.cat([self.embedding_net(embeddings), self.mist_net(mist_logits)], dim=1)
        residual = self.output(features)
        return mist_logits + self.residual_scale * residual


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a DreaMS-conditioned residual adapter over MIST logits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-embeddings-npz", required=True)
    parser.add_argument("--train-embeddings-key", default="embeddings")
    parser.add_argument("--train-embeddings-metadata", required=True)
    parser.add_argument("--train-mist-npz", required=True)
    parser.add_argument("--train-mist-key", default="probs")
    parser.add_argument("--train-mist-metadata", required=True)
    parser.add_argument("--train-targets-npz", required=True)
    parser.add_argument("--train-targets-key", default="ground_truth")
    parser.add_argument("--train-targets-metadata", required=True)

    parser.add_argument("--val-embeddings-npz", required=True)
    parser.add_argument("--val-embeddings-key", default="embeddings")
    parser.add_argument("--val-embeddings-metadata", required=True)
    parser.add_argument("--val-mist-npz", required=True)
    parser.add_argument("--val-mist-key", default="probs")
    parser.add_argument("--val-mist-metadata", required=True)
    parser.add_argument("--val-targets-npz", required=True)
    parser.add_argument("--val-targets-key", default="ground_truth")
    parser.add_argument("--val-targets-metadata", required=True)

    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--residual-scale", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=0.0)
    parser.add_argument("--disable-pos-weight", action="store_true")
    parser.add_argument("--soft-tanimoto-weight", type=float, default=0.0)
    parser.add_argument("--count-loss-weight", type=float, default=0.0)
    parser.add_argument("--mist-anchor-weight", type=float, default=0.0)
    parser.add_argument("--residual-l2-weight", type=float, default=0.0)
    parser.add_argument("--thresholds", default="0.1,0.125,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6")
    parser.add_argument("--top-k", default="32,48,64,80,96,128,160")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def clipped_logit(probs, eps=1e-5):
    probs = np.clip(np.asarray(probs, dtype=np.float32), eps, 1.0 - eps)
    return np.log(probs / (1.0 - probs)).astype(np.float32)


def spec_names_from_metadata(path):
    metadata = pd.read_csv(path)
    if "spec_name" not in metadata.columns:
        raise ValueError(f"Missing spec_name column in {path}")
    return [str(spec_name) for spec_name in metadata["spec_name"]]


def load_aligned_array(npz_path, key, metadata_path, spec_names):
    values = load_npz_array(npz_path, key).astype(np.float32)
    metadata = pd.read_csv(metadata_path)
    if "spec_name" not in metadata.columns:
        raise ValueError(f"Missing spec_name column in {metadata_path}")
    index = {str(spec_name): i for i, spec_name in enumerate(metadata["spec_name"])}
    missing = [spec_name for spec_name in spec_names if spec_name not in index]
    if missing:
        raise ValueError(f"{len(missing)} spec_names missing from {metadata_path}: {missing[:5]}")
    return values[[index[spec_name] for spec_name in spec_names]]


def load_split(embeddings_npz, embeddings_key, embeddings_metadata, mist_npz, mist_key, mist_metadata, targets_npz, targets_key, targets_metadata):
    spec_names = spec_names_from_metadata(targets_metadata)
    embeddings = load_aligned_array(embeddings_npz, embeddings_key, embeddings_metadata, spec_names)
    mist_probs = load_aligned_array(mist_npz, mist_key, mist_metadata, spec_names)
    targets = load_npz_array(targets_npz, targets_key).astype(np.float32)
    if len(targets) != len(spec_names):
        raise ValueError(f"Targets rows ({len(targets)}) do not match metadata rows ({len(spec_names)})")
    if embeddings.shape[0] != mist_probs.shape[0] or mist_probs.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: embeddings={embeddings.shape}, mist={mist_probs.shape}, targets={targets.shape}"
        )
    return spec_names, embeddings, mist_probs, targets


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


def mist_anchor_loss(logits, mist_probs):
    return nn.functional.binary_cross_entropy_with_logits(logits, mist_probs)


def sweep_probs(probs, targets, thresholds, top_ks):
    target_binary = targets > 0.5
    target_counts = target_binary.sum(axis=1)
    rows = []
    for threshold in thresholds:
        rows.append(
            {
                "mode": "threshold",
                "value": threshold,
                **threshold_metrics(probs, target_binary, target_counts, threshold),
            }
        )
    topk_rows = topk_metrics(probs, target_binary, target_counts, top_ks)
    for k in top_ks:
        rows.append({"mode": "top_k", "value": k, **topk_rows[k]})
    return sorted(rows, key=lambda row: row["mean_tanimoto"], reverse=True)


def evaluate(model, loader, device, targets_np, thresholds, top_ks):
    model.eval()
    probs = []
    with torch.no_grad():
        for embeddings, mist_logits, _targets in loader:
            logits = model(embeddings.to(device), mist_logits.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    probs_np = np.vstack(probs).astype(np.float32)
    rows = sweep_probs(probs_np, targets_np, thresholds, top_ks)
    return rows, probs_np


def write_rows_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and aligning train split...", flush=True)
    train_spec_names, train_embeddings, train_mist_probs, train_targets = load_split(
        args.train_embeddings_npz,
        args.train_embeddings_key,
        args.train_embeddings_metadata,
        args.train_mist_npz,
        args.train_mist_key,
        args.train_mist_metadata,
        args.train_targets_npz,
        args.train_targets_key,
        args.train_targets_metadata,
    )
    print("Loading and aligning validation split...", flush=True)
    val_spec_names, val_embeddings, val_mist_probs, val_targets = load_split(
        args.val_embeddings_npz,
        args.val_embeddings_key,
        args.val_embeddings_metadata,
        args.val_mist_npz,
        args.val_mist_key,
        args.val_mist_metadata,
        args.val_targets_npz,
        args.val_targets_key,
        args.val_targets_metadata,
    )

    train_mist_logits = clipped_logit(train_mist_probs)
    val_mist_logits = clipped_logit(val_mist_probs)
    thresholds = parse_number_list(args.thresholds, float)
    top_ks = parse_number_list(args.top_k, int)

    baseline_rows = sweep_probs(val_mist_probs, val_targets, thresholds, top_ks)
    (output_dir / "mist_baseline_metrics.json").write_text(json.dumps({"rows": baseline_rows}, indent=2))
    write_rows_csv(output_dir / "mist_baseline_metrics.csv", baseline_rows)
    print("Best MIST baseline:", json.dumps(baseline_rows[0], indent=2), flush=True)

    train_dataset = TensorDataset(
        torch.as_tensor(train_embeddings, dtype=torch.float32),
        torch.as_tensor(train_mist_logits, dtype=torch.float32),
        torch.as_tensor(train_mist_probs, dtype=torch.float32),
        torch.as_tensor(train_targets, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.as_tensor(val_embeddings, dtype=torch.float32),
        torch.as_tensor(val_mist_logits, dtype=torch.float32),
        torch.as_tensor(val_targets, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
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
    model = MistDreamsResidualAdapter(
        embedding_dim=train_embeddings.shape[1],
        fingerprint_bits=train_targets.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
    ).to(device)

    if args.disable_pos_weight:
        criterion = nn.BCEWithLogitsLoss()
        pos_weight_value = None
    else:
        if args.pos_weight > 0:
            pos_weight_value = float(args.pos_weight)
        else:
            positives = float(train_targets.sum())
            negatives = float(train_targets.size - positives)
            pos_weight_value = negatives / max(positives, 1.0)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = {
        "epoch": 0,
        "best_metrics": baseline_rows[0],
        "best_delta_vs_mist": 0.0,
    }
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        bce_losses = []
        tan_losses = []
        count_losses = []
        anchor_losses = []
        residual_l2_losses = []
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for embeddings, mist_logits, mist_probs, targets in progress:
            embeddings = embeddings.to(device, non_blocking=True)
            mist_logits = mist_logits.to(device, non_blocking=True)
            mist_probs = mist_probs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(embeddings, mist_logits)
            bce = criterion(logits, targets)
            tan = soft_tanimoto_loss(logits, targets) if args.soft_tanimoto_weight else logits.new_tensor(0.0)
            count = active_count_loss(logits, targets) if args.count_loss_weight else logits.new_tensor(0.0)
            anchor = mist_anchor_loss(logits, mist_probs) if args.mist_anchor_weight else logits.new_tensor(0.0)
            residual_l2 = (
                torch.square(logits - mist_logits).mean()
                if args.residual_l2_weight
                else logits.new_tensor(0.0)
            )
            loss = (
                bce
                + args.soft_tanimoto_weight * tan
                + args.count_loss_weight * count
                + args.mist_anchor_weight * anchor
                + args.residual_l2_weight * residual_l2
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(float(loss.item()))
            bce_losses.append(float(bce.item()))
            tan_losses.append(float(tan.item()))
            count_losses.append(float(count.item()))
            anchor_losses.append(float(anchor.item()))
            residual_l2_losses.append(float(residual_l2.item()))
            progress.set_postfix(loss=f"{np.mean(losses[-20:]):.4f}")

        rows, val_probs = evaluate(model, val_loader, device, val_targets, thresholds, top_ks)
        epoch_best = rows[0]
        delta = epoch_best["mean_tanimoto"] - baseline_rows[0]["mean_tanimoto"]
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_bce_loss": float(np.mean(bce_losses)),
            "train_soft_tanimoto_loss": float(np.mean(tan_losses)),
            "train_count_loss": float(np.mean(count_losses)),
            "train_mist_anchor_loss": float(np.mean(anchor_losses)),
            "train_residual_l2_loss": float(np.mean(residual_l2_losses)),
            "best_val": epoch_best,
            "delta_vs_mist": float(delta),
        }
        history.append(epoch_record)
        (output_dir / "training_metrics.json").write_text(
            json.dumps(
                {
                    "args": vars(args),
                    "device": str(device),
                    "pos_weight": pos_weight_value,
                    "train_rows": len(train_spec_names),
                    "val_rows": len(val_spec_names),
                    "mist_baseline_best": baseline_rows[0],
                    "best": best,
                    "history": history,
                },
                indent=2,
            )
        )
        print(json.dumps(epoch_record, indent=2), flush=True)

        torch.save(
            {
                "state_dict": model.state_dict(),
                "args": vars(args),
                "embedding_dim": int(train_embeddings.shape[1]),
                "fingerprint_bits": int(train_targets.shape[1]),
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "residual_scale": args.residual_scale,
                "epoch": epoch,
                "metrics": epoch_record,
                "mist_baseline_best": baseline_rows[0],
                "pos_weight": pos_weight_value,
            },
            output_dir / "last_model.pt",
        )

        if epoch_best["mean_tanimoto"] > best["best_metrics"]["mean_tanimoto"]:
            best = {
                "epoch": epoch,
                "best_metrics": epoch_best,
                "best_delta_vs_mist": float(delta),
            }
            np.savez_compressed(output_dir / "val_predictions.npz", probs=val_probs)
            (output_dir / "val_prediction_metadata.csv").write_text(Path(args.val_targets_metadata).read_text())
            (output_dir / "val_sweep_metrics.json").write_text(json.dumps({"rows": rows}, indent=2))
            write_rows_csv(output_dir / "val_sweep_metrics.csv", rows)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "args": vars(args),
                    "embedding_dim": int(train_embeddings.shape[1]),
                    "fingerprint_bits": int(train_targets.shape[1]),
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "residual_scale": args.residual_scale,
                    "epoch": epoch,
                    "metrics": epoch_record,
                    "mist_baseline_best": baseline_rows[0],
                    "pos_weight": pos_weight_value,
                },
                output_dir / "best_model.pt",
            )

    summary = {
        "train_rows": int(len(train_spec_names)),
        "val_rows": int(len(val_spec_names)),
        "fingerprint_bits": int(train_targets.shape[1]),
        "embedding_dim": int(train_embeddings.shape[1]),
        "mist_baseline_best": baseline_rows[0],
        "best": best,
        "passed_encoder_gate": bool(best["best_delta_vs_mist"] >= 0.005),
        "gate_delta": 0.005,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
