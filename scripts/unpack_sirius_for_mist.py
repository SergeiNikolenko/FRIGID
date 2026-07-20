#!/usr/bin/env python3
"""Unpack SIRIUS 5 project-space trees into the layout expected by MIST."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path


def unpack_project(project_dir: Path, labels_path: Path) -> list[dict[str, str]]:
    with labels_path.open(newline="") as handle:
        expected = {
            row["spec"]: {
                "formula": row["formula"],
                "ionization": row["ionization"].replace(" ", ""),
            }
            for row in csv.DictReader(handle, delimiter="\t")
        }
    rows: list[dict[str, str]] = []
    for compound_dir in sorted(path for path in project_dir.iterdir() if path.is_dir()):
        if "_" not in compound_dir.name:
            continue
        _, spectrum_id = compound_dir.name.split("_", 1)
        for name in ("scores", "spectra", "trees"):
            archive = compound_dir / name
            extracted = compound_dir / f"{name}.unpacked"
            if archive.is_file():
                with zipfile.ZipFile(archive) as bundle:
                    bundle.extractall(extracted)
                archive.replace(compound_dir / f"{name}.zip")
                extracted.replace(archive)
            elif not archive.is_dir():
                raise FileNotFoundError(f"Missing SIRIUS {name} for {spectrum_id}")
        trees = list((compound_dir / "trees").glob("*.json"))
        spectra = list((compound_dir / "spectra").glob("*.tsv"))
        scores = list((compound_dir / "scores").glob("*.info"))
        if len(trees) != 1 or len(spectra) != 1 or len(scores) != 1:
            raise ValueError(f"Expected one forced-formula tree for {spectrum_id}")
        tree = json.loads(trees[0].read_text())
        annotations = tree["annotations"]
        observed_formula = tree["molecularFormula"]
        observed_adduct = annotations["precursorIonType"].replace(" ", "")
        if spectrum_id not in expected:
            raise ValueError(f"Unexpected SIRIUS spectrum ID {spectrum_id}")
        if observed_formula != expected[spectrum_id]["formula"]:
            raise ValueError(
                f"SIRIUS formula mismatch for {spectrum_id}: "
                f"expected={expected[spectrum_id]['formula']} "
                f"observed={observed_formula}"
            )
        if observed_adduct != expected[spectrum_id]["ionization"]:
            raise ValueError(
                f"SIRIUS adduct mismatch for {spectrum_id}: "
                f"expected={expected[spectrum_id]['ionization']} "
                f"observed={observed_adduct}"
            )
        info = {}
        for line in (compound_dir / "compound.info").read_text().splitlines():
            if "\t" in line:
                key, value = line.split("\t", 1)
                info[key] = value
        rows.append(
            {
                "spec_name": spectrum_id,
                "spec_file": str(spectra[0].resolve()),
                "tree_file": str(trees[0].resolve()),
                "adduct": observed_adduct,
                "pred_formula": observed_formula,
                "parentmass": info["ionMass"],
            }
        )
    observed = {row["spec_name"] for row in rows}
    if observed != set(expected):
        raise ValueError(
            f"SIRIUS/labels ID mismatch: missing={sorted(set(expected)-observed)[:5]}, "
            f"extra={sorted(observed-set(expected))[:5]}"
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write_summary(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "",
                "spec_name",
                "spec_file",
                "tree_file",
                "adduct",
                "pred_formula",
                "parentmass",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows({"": index, **row} for index, row in enumerate(rows))


def main() -> None:
    args = parse_args()
    rows = unpack_project(args.project_dir, args.labels)
    write_summary(rows, args.output)
    print(f"Prepared {len(rows)} SIRIUS trees for MIST")


if __name__ == "__main__":
    main()
