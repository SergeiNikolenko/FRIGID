#!/usr/bin/env python
"""Evaluate predicted fingerprint probabilities against Morgan targets."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from frigid.dreams_fingerprint_head import compute_fingerprint_metrics, load_npz_array  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate spectrum encoder fingerprint probabilities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions-npz", required=True)
    parser.add_argument("--predictions-key", default="probs")
    parser.add_argument("--targets-npz", required=True)
    parser.add_argument("--targets-key", default="ground_truth")
    parser.add_argument("--metadata-csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    probs = load_npz_array(args.predictions_npz, args.predictions_key).astype(np.float32)
    targets = load_npz_array(args.targets_npz, args.targets_key).astype(np.float32)
    metrics, tanimoto = compute_fingerprint_metrics(probs, targets, args.threshold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "fingerprint_metrics.json"
    per_case_path = output_dir / "fingerprint_per_case.csv"

    summary_path.write_text(
        json.dumps(
            {
                "rows": int(len(probs)),
                "fingerprint_bits": int(probs.shape[1]),
                "threshold": args.threshold,
                **metrics.__dict__,
            },
            indent=2,
        )
    )

    per_case = pd.DataFrame({"fingerprint_tanimoto": tanimoto})
    if args.metadata_csv:
        metadata = pd.read_csv(args.metadata_csv)
        if len(metadata) != len(per_case):
            raise ValueError(
                f"Metadata rows ({len(metadata)}) do not match predictions ({len(per_case)})"
            )
        per_case = pd.concat([metadata.reset_index(drop=True), per_case], axis=1)
    per_case.to_csv(per_case_path, index=False)

    print(f"Saved metrics: {summary_path}")
    print(f"Saved per-case metrics: {per_case_path}")
    print(json.dumps(metrics.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
