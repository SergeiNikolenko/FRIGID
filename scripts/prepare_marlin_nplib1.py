#!/usr/bin/env python3
# ruff: noqa: E402
"""Prepare row-locked NPLIB1 splits for MARLIN encoder and decoder evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from benchmark_spec2mol import load_config, load_spec_data
from dlm.utils.benchmark_utils import compute_morgan_fingerprint, get_inchikey_first_block


def peak_array(spectrum) -> np.ndarray:
    raw = spectrum.get_spec()
    if isinstance(raw, np.ndarray):
        peaks = raw
    else:
        arrays = [item if isinstance(item, np.ndarray) else item[1] for item in raw]
        peaks = np.vstack([array for array in arrays if len(array)])
    peaks = np.asarray(peaks, dtype=np.float32)
    if peaks.ndim != 2 or peaks.shape[1] != 2:
        raise ValueError(f"invalid peak shape: {peaks.shape}")
    return peaks


def mgf_record(name: str, precursor_mz: float, peaks: np.ndarray) -> str:
    lines = ["BEGIN IONS", f"TITLE={name}", f"SCANS={name}", f"PEPMASS={precursor_mz}"]
    lines.extend(f"{float(mz):.6f} {float(intensity):.8f}" for mz, intensity in peaks)
    lines.append("END IONS")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/spec2mol_benchmark_canopus.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-spectra", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = Path(args.data_dir)
    config["data"].update(
        {
            "datadir": str(data_dir),
            "labels_file": str(data_dir / "labels.tsv"),
            "split_file": str(data_dir / "splits" / "canopus_hplus_100_0.tsv"),
            "spec_folder": str(data_dir / "spec_files"),
            "subform_folder": str(data_dir / "subformulae" / "subformulae_default"),
        }
    )
    config["evaluation"]["split"] = args.split
    _, split_data = load_spec_data(config["data"], config["mist_encoder"], args.split, shuffle=False)
    if args.max_spectra:
        split_data = split_data[: args.max_spectra]

    rows = []
    fingerprints = []
    records = []
    for source_index, (spectrum, molecule) in enumerate(split_data):
        smiles = molecule.get_smiles()
        fingerprint = compute_morgan_fingerprint(smiles, 4096, 2)
        if fingerprint is None:
            continue
        metadata = spectrum.get_meta()
        precursor_mz = float(spectrum.parentmass or metadata.get("PEPMASS", 0) or 0)
        name = spectrum.get_spec_name()
        inchikey = molecule.get_inchikey()
        rows.append(
            {
                "fingerprint_index": len(rows),
                "source_index": source_index,
                "split": args.split,
                "spec_name": name,
                "precursor_mz": precursor_mz,
                "neutral_mass": precursor_mz - 1.007276466621,
                "smiles": smiles,
                "inchikey": inchikey,
                "inchikey_first_block": get_inchikey_first_block(inchikey),
            }
        )
        fingerprints.append(fingerprint.astype(np.uint8))
        records.append(mgf_record(name, precursor_mz, peak_array(spectrum)))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "metadata.csv"
    fingerprint_path = args.output_dir / "fingerprints.npz"
    mgf_path = args.output_dir / "spectra.mgf"
    pd.DataFrame(rows).to_csv(metadata_path, index=False)
    np.savez_compressed(fingerprint_path, ground_truth=np.stack(fingerprints))
    mgf_path.write_text("\n\n".join(records) + "\n")
    summary = {
        "split": args.split,
        "rows": len(rows),
        "metadata": str(metadata_path),
        "fingerprints": str(fingerprint_path),
        "mgf": str(mgf_path),
        "morgan_radius": 2,
        "morgan_bits": 4096,
        "neutral_mass_assumption": "[M+H]+: precursor_mz - proton_mass",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
