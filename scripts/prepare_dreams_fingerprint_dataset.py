#!/usr/bin/env python
"""Prepare FRIGID spectra and Morgan targets for DreaMS fingerprint-head work.

This script does not import DreaMS. It creates an offline handoff package:

- ``spectra.mgf`` for embedding extraction in a DreaMS environment;
- ``metadata.csv`` with spectrum IDs, split, formula, SMILES, and row order;
- ``fingerprints.npz`` with ground-truth Morgan fingerprints.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import RDLogger
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from benchmark_spec2mol import load_config, load_spec_data, merge_config_with_args  # noqa: E402
from dlm.utils.benchmark_utils import compute_morgan_fingerprint, get_inchikey_first_block  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare an offline DreaMS fingerprint-head dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/spec2mol_benchmark_msg.yaml")
    parser.add_argument("--data-dir", help="Spec data directory")
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--max-spectra", type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fp-bits", type=int, default=4096)
    parser.add_argument("--fp-radius", type=int, default=2)

    # Suppressed args keep compatibility with merge_config_with_args.
    parser.add_argument("--mist-checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dlm-checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fp-threshold", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--softmax-temp", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--randomness", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--formula-matches", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-attempts", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir-unused", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def mgf_entry(spec_name: str, formula: str, precursor_mz: float, peaks: np.ndarray) -> str:
    rows = [
        "BEGIN IONS",
        f"TITLE={spec_name}",
        f"SCANS={spec_name}",
        f"PEPMASS={precursor_mz}",
        f"FORMULA={formula}",
    ]
    rows.extend(f"{float(mz):.6f} {float(intensity):.8f}" for mz, intensity in peaks)
    rows.append("END IONS")
    return "\n".join(rows)


def main():
    args = parse_args()
    config = merge_config_with_args(load_config(args.config), args)
    config["evaluation"]["split"] = args.split

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, split_data = load_spec_data(
        config["data"],
        config["mist_encoder"],
        args.split,
        shuffle=False,
    )

    rows = []
    fingerprints = []
    mgf_entries = []
    count = min(len(split_data), args.max_spectra) if args.max_spectra else len(split_data)

    for index, (spec, mol) in enumerate(tqdm(split_data[:count], desc="Preparing spectra")):
        smiles = mol.get_smiles()
        fingerprint = compute_morgan_fingerprint(smiles, args.fp_bits, args.fp_radius)
        if fingerprint is None:
            print(f"Skipping {spec.get_spec_name()}: cannot compute Morgan fingerprint")
            continue

        meta = spec.get_meta()
        spectra = spec.get_spec()
        peaks = np.vstack([peak_array for _, peak_array in spectra if len(peak_array)])
        precursor_mz = float(spec.parentmass or meta.get("PEPMASS", 0) or 0)
        spec_name = spec.get_spec_name()
        formula = spec.get_spectra_formula()

        rows.append(
            {
                "fingerprint_index": len(rows),
                "source_index": index,
                "split": args.split,
                "spec_name": spec_name,
                "formula": formula,
                "precursor_mz": precursor_mz,
                "smiles": smiles,
                "inchi_key": mol.get_inchikey(),
                "inchi_key_first_block": get_inchikey_first_block(mol.get_inchikey()),
            }
        )
        fingerprints.append(fingerprint.astype(np.float32))
        mgf_entries.append(mgf_entry(spec_name, formula, precursor_mz, peaks))

    metadata_path = output_dir / "metadata.csv"
    fingerprints_path = output_dir / "fingerprints.npz"
    mgf_path = output_dir / "spectra.mgf"
    summary_path = output_dir / "summary.json"

    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    np.savez_compressed(fingerprints_path, ground_truth=np.stack(fingerprints))
    mgf_path.write_text("\n\n".join(mgf_entries) + "\n")
    summary_path.write_text(
        json.dumps(
            {
                "split": args.split,
                "rows": len(rows),
                "fp_bits": args.fp_bits,
                "fp_radius": args.fp_radius,
                "metadata_csv": str(metadata_path),
                "fingerprints_npz": str(fingerprints_path),
                "spectra_mgf": str(mgf_path),
            },
            indent=2,
        )
    )

    print(f"Saved metadata: {metadata_path}")
    print(f"Saved fingerprints: {fingerprints_path}")
    print(f"Saved MGF: {mgf_path}")
    print(f"Rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
