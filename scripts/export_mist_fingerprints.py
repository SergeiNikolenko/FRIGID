#!/usr/bin/env python
"""Export MIST fingerprint probabilities for a benchmark split."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
scripts_path = os.path.join(project_root, "scripts")
for path in (src_path, scripts_path):
    if path not in sys.path:
        sys.path.insert(0, path)

from benchmark_spec2mol import load_config, load_spec_data  # noqa: E402
from dlm.utils.benchmark_utils import (  # noqa: E402
    binarize_fingerprint,
    compute_morgan_fingerprint,
    get_inchikey_first_block,
)
from dlm.utils.utils_chem import smiles_to_safe  # noqa: E402
from frigid.fingerprint_backends import build_fingerprint_backend  # noqa: E402
from mist.data.datasets import get_paired_loader  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export MIST spectrum-to-fingerprint probabilities.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--mist-checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-spectra", type=int, default=None)
    parser.add_argument("--array-key", default="probs")
    parser.add_argument("--fp-threshold", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dlm-checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--softmax-temp", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--randomness", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--formula-matches", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-attempts", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def apply_overrides(config, args):
    config["mist_encoder"]["checkpoint"] = args.mist_checkpoint
    config.setdefault("encoder_backend", {})["name"] = "mist"
    config["data"]["datadir"] = args.data_dir
    config["data"]["labels_file"] = os.path.join(args.data_dir, "labels.tsv")
    config["data"]["split_file"] = os.path.join(args.data_dir, "split.tsv")
    config["data"]["spec_folder"] = os.path.join(args.data_dir, "spec_files")
    config["data"]["subform_folder"] = os.path.join(
        args.data_dir, "subformulae/default_subformulae"
    )
    config["evaluation"]["split"] = args.split
    if args.fp_threshold is not None:
        config["fingerprint"]["threshold"] = args.fp_threshold
    return config


def main():
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, split_data = load_spec_data(
        config["data"],
        config["mist_encoder"],
        config["evaluation"]["split"],
        shuffle=False,
    )
    fingerprint_backend = build_fingerprint_backend(config, device)
    loader = get_paired_loader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=0,
    )

    probs = []
    binary = []
    ground_truth = []
    rows = []
    max_spectra = args.max_spectra or len(dataset)
    fp_cfg = config["fingerprint"]
    fp_bits = fp_cfg["bits"]
    fp_radius = fp_cfg["radius"]
    fp_threshold = fp_cfg["threshold"]
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, total=min(len(dataset), max_spectra))):
            if idx >= max_spectra:
                break
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            spec, mol = split_data[idx]
            smiles = mol.get_smiles()
            gt_fp = compute_morgan_fingerprint(smiles, fp_bits, fp_radius)
            if gt_fp is None:
                print(f"Warning: Could not compute FP for {smiles}, skipping")
                continue
            pred = fingerprint_backend.predict_probs(batch, spec_name=spec.get_spec_name())
            pred_binary = binarize_fingerprint(pred, fp_threshold).astype(np.float32)
            probs.append(np.asarray(pred, dtype=np.float32))
            binary.append(pred_binary)
            ground_truth.append(gt_fp.astype(np.float32))
            rows.append(
                {
                    "row_index": idx,
                    "fingerprint_index": len(rows),
                    "split": args.split,
                    "spec_name": spec.get_spec_name(),
                    "smiles": smiles,
                    "input": smiles_to_safe(smiles),
                    "inchi_key": mol.get_inchikey(),
                    "inchikey": mol.get_inchikey(),
                    "inchi_key_first_block": get_inchikey_first_block(mol.get_inchikey()),
                }
            )

    probs_np = np.vstack(probs).astype(np.float32)
    binary_np = np.vstack(binary).astype(np.float32)
    ground_truth_np = np.vstack(ground_truth).astype(np.float32)
    predictions_path = output_dir / "fingerprints.npz"
    metadata_path = output_dir / "metadata.csv"
    summary_path = output_dir / "summary.json"
    arrays = {
        args.array_key: probs_np,
        "mist_probs": probs_np,
        "mist_binary": binary_np,
        "ground_truth": ground_truth_np,
    }
    np.savez_compressed(predictions_path, **arrays)
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "split": args.split,
                "rows": int(len(rows)),
                "fingerprint_bits": int(probs_np.shape[1]),
                "array_key": args.array_key,
                "compatibility_keys": ["mist_probs", "mist_binary", "ground_truth"],
                "fp_threshold": fp_threshold,
                "checkpoint": args.mist_checkpoint,
                "data_dir": args.data_dir,
                "predictions_npz": str(predictions_path),
                "metadata_csv": str(metadata_path),
            },
            indent=2,
        )
    )
    print(f"Saved probabilities: {predictions_path}")
    print(f"Saved metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
