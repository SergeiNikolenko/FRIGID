#!/usr/bin/env python
"""Build the exact MIST-compatible MSG row manifest for frozen probes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from frigid.encoder_benchmark import sha256_file  # noqa: E402


SUPPORTED_MIST_ATOMS = frozenset({"C", "O", "P", "N", "S", "Cl", "F", "H"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a row-locked MSG metadata manifest for frozen probes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary")
    return parser.parse_args()


def supported_mist_structure(smiles: str) -> tuple[bool, set[str]]:
    """Return whether a structure matches the atom domain used by MIST loading."""

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return False, {"invalid_smiles"}
    atom_symbols = {atom.GetSymbol() for atom in molecule.GetAtoms()}
    unsupported = atom_symbols - SUPPORTED_MIST_ATOMS
    return not unsupported, unsupported


def build_metadata(
    labels_path: str | Path,
    split_path: str | Path,
    *,
    split: str,
) -> tuple[pd.DataFrame, dict]:
    """Join MSG labels/splits and reproduce MIST's supported-atom filter."""

    labels = pd.read_csv(labels_path, sep="\t", dtype=str)
    splits = pd.read_csv(split_path, sep="\t", dtype=str)
    required_labels = {"spec", "smiles", "inchikey"}
    required_splits = {"name", "split"}
    if missing := sorted(required_labels - set(labels.columns)):
        raise ValueError(f"Labels are missing columns: {missing}")
    if missing := sorted(required_splits - set(splits.columns)):
        raise ValueError(f"Split file is missing columns: {missing}")
    if labels["spec"].duplicated().any() or splits["name"].duplicated().any():
        raise ValueError("Labels and split files must contain unique spectrum IDs")

    joined = labels.merge(
        splits.rename(columns={"name": "spec"}),
        on="spec",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(labels) or len(joined) != len(splits):
        raise ValueError("Labels and split files do not contain the same spectrum IDs")
    selected = joined.loc[joined["split"].eq(split)].copy()
    if selected.empty:
        raise ValueError(f"Split {split!r} is empty")

    keep = []
    unsupported_counts: Counter[str] = Counter()
    for smiles in selected["smiles"]:
        supported, unsupported = supported_mist_structure(smiles)
        keep.append(supported)
        unsupported_counts.update(unsupported)
    eligible = selected.loc[keep].copy().reset_index(drop=True)
    eligible.insert(0, "fingerprint_index", range(len(eligible)))
    eligible = eligible.rename(
        columns={"spec": "spec_name", "inchikey": "inchi_key"}
    )
    eligible["inchi_key"] = eligible["inchi_key"].astype(str).str.strip()
    eligible["inchi_key_first_block"] = (
        eligible["inchi_key"].str.split("-", n=1).str[0].str.upper()
    )
    ordered_columns = [
        "fingerprint_index",
        "split",
        "spec_name",
        "smiles",
        "inchi_key",
        "inchi_key_first_block",
    ]
    ordered_columns.extend(
        column for column in eligible.columns if column not in ordered_columns
    )
    eligible = eligible[ordered_columns]
    summary = {
        "split": split,
        "source_rows": int(len(selected)),
        "eligible_rows": int(len(eligible)),
        "filtered_rows": int(len(selected) - len(eligible)),
        "supported_atoms": sorted(SUPPORTED_MIST_ATOMS),
        "unsupported_atom_counts": dict(sorted(unsupported_counts.items())),
        "unique_structures": int(eligible["inchi_key_first_block"].nunique()),
    }
    return eligible, summary


def main() -> int:
    args = parse_args()
    labels_path = Path(args.labels).resolve()
    split_path = Path(args.split_file).resolve()
    output_path = Path(args.output).resolve()
    summary_path = Path(args.summary or output_path.with_suffix(".summary.json")).resolve()
    for path in (output_path, summary_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    RDLogger.DisableLog("rdApp.*")
    metadata, summary = build_metadata(labels_path, split_path, split=args.split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_path, index=False)
    summary.update(
        {
            "labels": str(labels_path),
            "labels_sha256": sha256_file(labels_path),
            "split_file": str(split_path),
            "split_file_sha256": sha256_file(split_path),
            "output": str(output_path),
            "output_sha256": sha256_file(output_path),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
