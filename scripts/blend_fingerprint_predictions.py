#!/usr/bin/env python
"""Evaluate alpha blends of two fingerprint probability sources."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
scripts_path = os.path.join(project_root, "scripts")
for path in (src_path, scripts_path):
    if path not in sys.path:
        sys.path.insert(0, path)

from frigid.dreams_fingerprint_head import compute_fingerprint_metrics, load_npz_array  # noqa: E402
from sweep_fingerprint_predictions import metrics_from_binary, topk_binary  # noqa: E402


def parse_number_list(value, cast):
    if not value:
        return []
    return [cast(item) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Blend two fingerprint probability arrays and sweep calibration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--primary-npz", required=True)
    parser.add_argument("--primary-metadata", required=True)
    parser.add_argument("--primary-key", default="probs")
    parser.add_argument("--secondary-npz", required=True)
    parser.add_argument("--secondary-metadata", required=True)
    parser.add_argument("--secondary-key", default="probs")
    parser.add_argument("--targets-npz", required=True)
    parser.add_argument("--targets-metadata", required=True)
    parser.add_argument("--targets-key", default="ground_truth")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alphas", default="0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,1.0")
    parser.add_argument(
        "--thresholds",
        default="0.02,0.03,0.04,0.05,0.075,0.1,0.125,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8",
    )
    parser.add_argument("--top-k", default="32,48,64,80,96,128,160,192,256,320")
    return parser.parse_args()


def load_aligned_array(npz_path, key, metadata_path, spec_names):
    values = load_npz_array(npz_path, key).astype(np.float32)
    metadata = pd.read_csv(metadata_path)
    if "spec_name" not in metadata.columns:
        raise ValueError(f"Missing spec_name column in {metadata_path}")
    index = {str(spec_name): i for i, spec_name in enumerate(metadata["spec_name"])}
    missing = [spec_name for spec_name in spec_names if spec_name not in index]
    if missing:
        raise ValueError(f"{len(missing)} target spec_names missing from {metadata_path}: {missing[:5]}")
    return values[[index[spec_name] for spec_name in spec_names]]


def main():
    args = parse_args()
    target_metadata = pd.read_csv(args.targets_metadata)
    if "spec_name" not in target_metadata.columns:
        raise ValueError(f"Missing spec_name column in {args.targets_metadata}")
    spec_names = [str(spec_name) for spec_name in target_metadata["spec_name"]]

    targets = load_npz_array(args.targets_npz, args.targets_key).astype(np.float32)
    if len(targets) != len(spec_names):
        raise ValueError(
            f"Targets rows ({len(targets)}) do not match metadata rows ({len(spec_names)})"
        )

    primary = load_aligned_array(args.primary_npz, args.primary_key, args.primary_metadata, spec_names)
    secondary = load_aligned_array(
        args.secondary_npz,
        args.secondary_key,
        args.secondary_metadata,
        spec_names,
    )
    if primary.shape != secondary.shape or primary.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: primary={primary.shape}, secondary={secondary.shape}, targets={targets.shape}"
        )

    rows = []
    for alpha in parse_number_list(args.alphas, float):
        blend = alpha * primary + (1.0 - alpha) * secondary
        for threshold in parse_number_list(args.thresholds, float):
            metrics, _ = compute_fingerprint_metrics(blend, targets, threshold)
            rows.append(
                {
                    "mode": "threshold",
                    "alpha_primary": alpha,
                    "value": threshold,
                    **metrics.__dict__,
                }
            )
        for k in parse_number_list(args.top_k, int):
            rows.append(
                {
                    "mode": "top_k",
                    "alpha_primary": alpha,
                    "value": k,
                    **metrics_from_binary(topk_binary(blend, k), targets),
                }
            )

    rows = sorted(rows, key=lambda row: row["mean_tanimoto"], reverse=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "blend_metrics.json").write_text(json.dumps({"rows": rows}, indent=2))

    csv_path = output_dir / "blend_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "rows": int(len(targets)),
        "fingerprint_bits": int(targets.shape[1]),
        "primary_npz": args.primary_npz,
        "secondary_npz": args.secondary_npz,
        "targets_npz": args.targets_npz,
        "best": rows[0],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(rows[:20], indent=2))
    print(f"Saved blend metrics: {output_dir / 'blend_metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
