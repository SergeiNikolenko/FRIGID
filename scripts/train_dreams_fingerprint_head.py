#!/usr/bin/env python
"""Train a fingerprint head on precomputed DreaMS embeddings."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from frigid.dreams_fingerprint_head import (  # noqa: E402
    FingerprintEmbeddingDataset,
    FingerprintHead,
    checkpoint_payload,
    compute_fingerprint_metrics,
    load_npz_array,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DreaMS embedding -> Morgan fingerprint head.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--embeddings-npz", required=True)
    parser.add_argument("--embeddings-key", default="embeddings")
    parser.add_argument("--fingerprints-npz", required=True)
    parser.add_argument("--fingerprints-key", default="ground_truth")
    parser.add_argument("--val-embeddings-npz")
    parser.add_argument("--val-fingerprints-npz")
    parser.add_argument("--val-embeddings-key", default=None)
    parser.add_argument("--val-fingerprints-key", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--pos-weight", type=float, default=0.0, help="Manual BCE positive class weight; 0 estimates it from train targets.")
    parser.add_argument("--disable-pos-weight", action="store_true", help="Use an unweighted BCE term even when --pos-weight is 0.")
    parser.add_argument("--soft-tanimoto-weight", type=float, default=0.0, help="Weight for an auxiliary differentiable Tanimoto loss.")
    parser.add_argument("--count-loss-weight", type=float, default=0.0, help="Weight for an auxiliary active-bit count loss.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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


def evaluate(model, loader, device, threshold):
    model.eval()
    losses = []
    probs = []
    targets = []
    criterion = torch.nn.BCEWithLogitsLoss(reduction="mean")
    with torch.no_grad():
        for embeddings, fingerprints in loader:
            embeddings = embeddings.to(device)
            fingerprints = fingerprints.to(device)
            logits = model(embeddings)
            losses.append(float(criterion(logits, fingerprints).item()))
            probs.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(fingerprints.cpu().numpy())
    probs_np = np.vstack(probs)
    targets_np = np.vstack(targets)
    metrics, _ = compute_fingerprint_metrics(probs_np, targets_np, threshold)
    return float(np.mean(losses)), metrics, probs_np


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_embeddings = load_npz_array(args.embeddings_npz, args.embeddings_key).astype(np.float32)
    train_fingerprints = load_npz_array(args.fingerprints_npz, args.fingerprints_key).astype(np.float32)
    train_dataset_full = FingerprintEmbeddingDataset(train_embeddings, train_fingerprints)

    explicit_val = bool(args.val_embeddings_npz or args.val_fingerprints_npz)
    if explicit_val and not (args.val_embeddings_npz and args.val_fingerprints_npz):
        raise ValueError("Provide both --val-embeddings-npz and --val-fingerprints-npz")

    if explicit_val:
        val_embeddings_key = args.val_embeddings_key or args.embeddings_key
        val_fingerprints_key = args.val_fingerprints_key or args.fingerprints_key
        val_embeddings = load_npz_array(args.val_embeddings_npz, val_embeddings_key).astype(np.float32)
        val_fingerprints = load_npz_array(args.val_fingerprints_npz, val_fingerprints_key).astype(np.float32)
        train_ds = train_dataset_full
        val_ds = FingerprintEmbeddingDataset(val_embeddings, val_fingerprints)
        train_size = len(train_ds)
        val_size = len(val_ds)
        embeddings = train_embeddings
        fingerprints = train_fingerprints
    else:
        dataset = train_dataset_full
        val_size = int(round(len(dataset) * args.val_fraction))
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(args.seed)
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
        embeddings = train_embeddings
        fingerprints = train_fingerprints

    if train_size <= 0:
        raise ValueError("Training set is empty after applying --val-fraction")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if val_size else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FingerprintHead(
        input_dim=embeddings.shape[1],
        fingerprint_bits=fingerprints.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    if hasattr(train_ds, "indices"):
        train_targets = fingerprints[train_ds.indices]
    else:
        train_targets = train_fingerprints
    if args.disable_pos_weight:
        pos_weight_value = 1.0
        pos_weight = None
    elif args.pos_weight > 0:
        pos_weight_value = args.pos_weight
        pos_weight = torch.full((fingerprints.shape[1],), pos_weight_value, device=device)
    else:
        positives = float(train_targets.sum())
        negatives = float(train_targets.size - positives)
        pos_weight_value = negatives / positives if positives > 0 else 1.0
        pos_weight = torch.full((fingerprints.shape[1],), pos_weight_value, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_metric = -1.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "dreams_fingerprint_head.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch_embeddings, batch_fps in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            batch_embeddings = batch_embeddings.to(device)
            batch_fps = batch_fps.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_embeddings)
            bce_loss = criterion(logits, batch_fps)
            loss = bce_loss
            if args.soft_tanimoto_weight:
                loss = loss + args.soft_tanimoto_weight * soft_tanimoto_loss(logits, batch_fps)
            if args.count_loss_weight:
                loss = loss + args.count_loss_weight * active_count_loss(logits, batch_fps)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        if val_loader is not None:
            val_loss, val_metrics, _ = evaluate(model, val_loader, device, args.threshold)
            metric_value = val_metrics.mean_tanimoto
            row = {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
                **val_metrics.__dict__,
            }
        else:
            val_metrics = None
            metric_value = -float(np.mean(train_losses))
            row = {"epoch": epoch, "train_loss": float(np.mean(train_losses))}
        history.append(row)

        if metric_value > best_metric:
            best_metric = metric_value
            torch.save(
                checkpoint_payload(
                    model,
                    input_dim=embeddings.shape[1],
                    fingerprint_bits=fingerprints.shape[1],
                    hidden_dim=args.hidden_dim,
                    dropout=args.dropout,
                    threshold=args.threshold,
                    metrics=row,
                ),
                best_path,
            )

    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "rows": train_size + val_size,
                "train_rows": train_size,
                "val_rows": val_size,
                "explicit_val": explicit_val,
                "embedding_dim": int(embeddings.shape[1]),
                "fingerprint_bits": int(fingerprints.shape[1]),
                "threshold": args.threshold,
                "pos_weight": pos_weight_value,
                "disable_pos_weight": args.disable_pos_weight,
                "soft_tanimoto_weight": args.soft_tanimoto_weight,
                "count_loss_weight": args.count_loss_weight,
                "history": history,
                "best_checkpoint": str(best_path),
            },
            indent=2,
        )
    )
    print(f"Saved checkpoint: {best_path}")
    print(f"Saved metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
