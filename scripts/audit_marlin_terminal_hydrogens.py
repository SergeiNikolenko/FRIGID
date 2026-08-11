#!/usr/bin/env python3
"""Audit the terminal EOS mass gate against every gold target of a split.

The gate that lets the decoder emit EOS accepts a finished SAFE string only when
its mass matches the conditioning mass. That gate has to accept every gold
target: if it rejects one, no candidate the decoder could ever produce for that
spectrum would be admitted, and the spectrum is lost before ranking.

For each SMILES this converts to SAFE exactly as training does, scans it with the
grammar, and asserts the gate accepts it at the mass the run conditions on for it,
which is its monoisotopic mass less a proton per unit of formal charge, within the
decode tolerance. It also reports how many hydrogen counts the gate admits per
target -- the width the exact terminal count collapses -- and how far the
scanner's hydrogen count sits from RDKit's, which is the defect this measures.

Usage:
  PYTHONPATH=src python scripts/audit_marlin_terminal_hydrogens.py \
      --metadata .../test/metadata.csv --metadata .../train/metadata.csv
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

import pandas as pd
from rdkit import Chem

from dlm.utils.utils_chem import smiles_to_safe
from marlin.grammar import (
    _has_hydrogen_only_exact_mass,
    _scan,
    _terminal_hydrogens,
)
from marlin.mass_shell import conditioning_mass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, action="append", required=True)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--ppm-tolerance", type=float, default=10.0)
    parser.add_argument("--valence-slack", type=float, default=4.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def audit_split(
    metadata: Path,
    smiles_column: str,
    ppm_tolerance: float,
    valence_slack: float,
) -> dict:
    table = pd.read_csv(metadata)
    if smiles_column not in table:
        raise KeyError(f"missing SMILES column {smiles_column!r} in {metadata}")
    rejections: list[dict] = []
    interval_widths: list[int] = []
    exact_widths: list[int] = []
    hydrogen_errors: list[int] = []
    scan_failures: list[dict] = []
    rows = 0

    for row_index, smiles in enumerate(table[smiles_column].astype(str)):
        safe = smiles_to_safe(smiles)
        molecule = None if safe is None else Chem.MolFromSmiles(safe)
        state = None if safe is None else _scan(safe)
        if molecule is None or state is None or not state.terminal:
            scan_failures.append({"row": row_index, "smiles": smiles, "safe": safe})
            continue
        rows += 1
        # The mass the gate is held to is the one the run conditions on, derived
        # from the precursor as precursor_mz - proton. For the 7,533 uncharged
        # targets that is ExactMolWt; for the 152 charged ones of train and test
        # it is a proton lower, and holding the mask to ExactMolWt instead is the
        # defect the conditioning mass fixes (see mass_shell.conditioning_mass).
        target_mass = conditioning_mass(molecule)
        tolerance = ppm_tolerance * 1e-6 * target_mass

        # The interval the gate used to admit, in hydrogen counts.
        minimum, maximum = state.hydrogen_bounds(valence_slack)
        interval_widths.append(maximum - minimum + 1)
        exact_widths.append(1)

        truth = sum(atom.GetTotalNumHs() for atom in molecule.GetAtoms())
        hydrogen_errors.append(_terminal_hydrogens(state) - truth)

        if not _has_hydrogen_only_exact_mass(
            state, target_mass, valence_slack, tolerance
        ):
            rejections.append(
                {
                    "row": row_index,
                    "smiles": smiles,
                    "safe": safe,
                    "target_mass": target_mass,
                    "scanner_hydrogens": _terminal_hydrogens(state),
                    "rdkit_hydrogens": truth,
                }
            )

    return {
        "metadata": str(metadata),
        "targets": rows,
        "scan_failures": len(scan_failures),
        "scan_failure_examples": scan_failures[:5],
        "gold_rejections": len(rejections),
        "rejection_examples": rejections[:10],
        "hydrogen_count_errors": {
            "exact": sum(1 for error in hydrogen_errors if error == 0),
            "wrong": sum(1 for error in hydrogen_errors if error != 0),
        },
        "median_gate_width_interval": statistics.median(interval_widths),
        "median_gate_width_exact": statistics.median(exact_widths),
        "max_gate_width_interval": max(interval_widths),
    }


def main() -> None:
    args = parse_args()
    splits = [
        audit_split(
            metadata,
            args.smiles_column,
            args.ppm_tolerance,
            args.valence_slack,
        )
        for metadata in args.metadata
    ]
    manifest = {
        "kind": "MARLIN terminal EOS mass gate audit",
        "git_commit": _git_commit(),
        "ppm_tolerance": args.ppm_tolerance,
        "valence_slack": args.valence_slack,
        "splits": splits,
        "totals": {
            "targets": sum(split["targets"] for split in splits),
            "gold_rejections": sum(split["gold_rejections"] for split in splits),
            "scan_failures": sum(split["scan_failures"] for split in splits),
            "hydrogen_count_wrong": sum(
                split["hydrogen_count_errors"]["wrong"] for split in splits
            ),
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
