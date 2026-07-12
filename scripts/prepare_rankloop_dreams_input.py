#!/usr/bin/env python
"""Prepare an ordered MGF handoff for frozen DreaMS RankLoop embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from benchmark_spec2mol import load_config, load_spec_data, merge_config_with_args  # noqa: E402
from dlm.utils.benchmark_selection import (  # noqa: E402
    load_spec_manifest,
    resolve_selected_indices,
)
from frigid.rankloop_corpus import _git_revision, sha256_file  # noqa: E402
from frigid.rankloop_dreams import (  # noqa: E402
    build_dreams_mgf_entry,
    merge_spectrum_peaks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare exact ordered spectra for frozen DreaMS inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/spec2mol_benchmark_msg.yaml")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--spec-manifest")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-spectra", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mist-checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dlm-checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fp-threshold", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--softmax-temp", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--randomness", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--formula-matches", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-attempts", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = merge_config_with_args(load_config(args.config), args)
    _dataset, split_data = load_spec_data(
        config["data"],
        config["mist_encoder"],
        config["evaluation"]["split"],
        shuffle=False,
    )
    manifest_names = (
        load_spec_manifest(args.spec_manifest) if args.spec_manifest else None
    )
    selected_indices = resolve_selected_indices(
        split_data,
        manifest_names,
        args.start_index,
        args.max_spectra,
    )

    rows: list[dict[str, object]] = []
    mgf_entries: list[str] = []
    for original_index in tqdm(selected_indices, desc="Preparing DreaMS spectra"):
        spectrum, molecule = split_data[original_index]
        peaks = merge_spectrum_peaks(spectrum.get_spec())
        precursor_mz = float(spectrum.parentmass)
        spec_name = spectrum.get_spec_name()
        formula = spectrum.get_spectra_formula()
        mgf_entries.append(
            build_dreams_mgf_entry(
                spec_name=spec_name,
                formula=formula,
                precursor_mz=precursor_mz,
                peaks=peaks,
            )
        )
        rows.append(
            {
                "embedding_index": len(rows),
                "source_index": original_index,
                "split": args.split,
                "spec_name": spec_name,
                "formula": formula,
                "precursor_mz": precursor_mz,
                "peak_count": len(peaks),
                "smiles": molecule.get_smiles(),
                "inchi_key": molecule.get_inchikey(),
                "instrument": spectrum.get_instrument(),
            }
        )
    if not rows:
        raise ValueError("No spectra were selected for DreaMS input preparation.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"
    mgf_path = output_dir / "spectra.mgf"
    metadata = pd.DataFrame(rows)
    metadata.to_csv(metadata_path, index=False)
    mgf_path.write_text("\n\n".join(mgf_entries) + "\n", encoding="utf-8")

    config_path = Path(args.config).expanduser().resolve()
    spec_manifest_path = (
        Path(args.spec_manifest).expanduser().resolve() if args.spec_manifest else None
    )
    labels_path = Path(config["data"]["labels_file"]).expanduser().resolve()
    split_path = Path(config["data"]["split_file"]).expanduser().resolve()
    revision, dirty = _git_revision(PROJECT_ROOT)
    run_manifest = {
        "schema_version": 1,
        "stage": "rankloop_dreams_input",
        "repo": {"commit": revision, "dirty": dirty},
        "parameters": vars(args),
        "inputs": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "labels": {"path": str(labels_path), "sha256": sha256_file(labels_path)},
            "split": {"path": str(split_path), "sha256": sha256_file(split_path)},
            "spec_manifest": (
                {
                    "path": str(spec_manifest_path),
                    "sha256": sha256_file(spec_manifest_path),
                }
                if spec_manifest_path
                else None
            ),
        },
        "outputs": {
            "metadata_csv": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "spectra_mgf": str(mgf_path),
            "spectra_mgf_sha256": sha256_file(mgf_path),
            "row_count": len(metadata),
            "total_peak_count": int(metadata["peak_count"].sum()),
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(run_manifest["outputs"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
