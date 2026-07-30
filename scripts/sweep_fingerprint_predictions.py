#!/usr/bin/env python
"""Sweep threshold and top-k calibration for fingerprint probabilities."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from frigid.dreams_fingerprint_head import (  # noqa: E402
    compute_fingerprint_metrics,
    load_npz_array,
    tanimoto_per_row,
)


def parse_number_list(value, cast):
    if not value:
        return []
    return [cast(item) for item in value.split(",") if item.strip()]


def metrics_from_binary(pred, target):
    target = (target > 0.5).astype(np.float32)
    pred = pred.astype(np.float32)
    tanimoto = tanimoto_per_row(pred, target)
    tp = float(np.logical_and(pred == 1, target == 1).sum())
    fp = float(np.logical_and(pred == 1, target == 0).sum())
    fn = float(np.logical_and(pred == 0, target == 1).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "mean_tanimoto": float(tanimoto.mean()) if len(tanimoto) else 0.0,
        "median_tanimoto": float(np.median(tanimoto)) if len(tanimoto) else 0.0,
        "bit_accuracy": float((pred == target).mean()),
        "bit_precision": precision,
        "bit_recall": recall,
        "bit_f1": f1,
        "mean_active_bits": float(pred.sum(axis=1).mean()) if len(pred) else 0.0,
        "mean_target_active_bits": float(target.sum(axis=1).mean()) if len(target) else 0.0,
    }


def topk_binary(probs, k):
    k = min(max(int(k), 1), probs.shape[1])
    indices = np.argpartition(probs, -k, axis=1)[:, -k:]
    pred = np.zeros_like(probs, dtype=np.float32)
    rows = np.arange(probs.shape[0])[:, None]
    pred[rows, indices] = 1.0
    return pred


def target_count_topk_binary(probs, targets):
    counts = np.maximum((targets > 0.5).sum(axis=1).astype(int), 1)
    pred = np.zeros_like(probs, dtype=np.float32)
    for row_idx, k in enumerate(counts):
        k = min(k, probs.shape[1])
        indices = np.argpartition(probs[row_idx], -k)[-k:]
        pred[row_idx, indices] = 1.0
    return pred


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep fingerprint probability calibration settings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions-npz", required=True)
    parser.add_argument("--predictions-key", default="probs")
    parser.add_argument("--targets-npz", required=True)
    parser.add_argument("--targets-key", default="ground_truth")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--thresholds",
        default="0.02,0.03,0.04,0.05,0.075,0.1,0.125,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8",
    )
    parser.add_argument(
        "--top-k",
        default="32,48,64,80,96,128,160,192,256,320",
        help="Comma-separated fixed top-k active-bit counts to evaluate.",
    )
    parser.add_argument("--include-target-count-topk", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    probs = load_npz_array(args.predictions_npz, args.predictions_key).astype(np.float32)
    targets = load_npz_array(args.targets_npz, args.targets_key).astype(np.float32)
    if probs.shape != targets.shape:
        raise ValueError(f"Predictions {probs.shape} and targets {targets.shape} differ")

    rows = []
    for threshold in parse_number_list(args.thresholds, float):
        metrics, _ = compute_fingerprint_metrics(probs, targets, threshold)
        rows.append({"mode": "threshold", "value": threshold, **metrics.__dict__})

    for k in parse_number_list(args.top_k, int):
        rows.append({"mode": "top_k", "value": k, **metrics_from_binary(topk_binary(probs, k), targets)})

    if args.include_target_count_topk:
        rows.append(
            {
                "mode": "target_count_top_k",
                "value": "target_active_bits",
                **metrics_from_binary(target_count_topk_binary(probs, targets), targets),
            }
        )

    rows = sorted(rows, key=lambda row: row["mean_tanimoto"], reverse=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sweep_metrics.json").write_text(json.dumps({"rows": rows}, indent=2))

    csv_path = output_dir / "sweep_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(rows[:10], indent=2))
    print(f"Saved sweep metrics: {output_dir / 'sweep_metrics.json'}")
    print(f"Saved sweep CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
