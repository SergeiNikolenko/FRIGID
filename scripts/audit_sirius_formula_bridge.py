#!/usr/bin/env python3
"""Audit SIRIUS formula/adduct preservation and request ranked fallbacks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path

from scripts.unpack_sirius_for_mist import (
    _expected_tree_formula,
    _read_ms_headers,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_info(path: Path) -> dict[str, str]:
    info = {}
    for line in path.read_text().splitlines():
        if "\t" in line:
            key, value = line.split("\t", 1)
            info[key] = value
    return info


def _read_single_tree(path: Path) -> dict:
    if path.is_dir():
        trees = list(path.glob("*.json"))
        if len(trees) != 1:
            raise ValueError(f"Expected one SIRIUS tree in {path}, got {len(trees)}")
        return json.loads(trees[0].read_text())
    if not path.is_file():
        raise FileNotFoundError(f"Missing SIRIUS trees archive {path}")
    with zipfile.ZipFile(path) as bundle:
        members = [name for name in bundle.namelist() if name.endswith(".json")]
        if len(members) != 1:
            raise ValueError(
                f"Expected one SIRIUS tree in {path}, got {len(members)}"
            )
        return json.loads(bundle.read(members[0]))


def audit_project(
    project_dir: Path,
    labels_path: Path,
    output_path: Path,
    manifest_path: Path | None = None,
) -> dict:
    with labels_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected = {row["spec"]: row for row in rows}
    if len(expected) != len(rows):
        raise ValueError("Labels contain duplicate spectrum IDs")

    evidence = []
    mismatches = []
    observed_ids = set()
    for compound_dir in sorted(path for path in project_dir.iterdir() if path.is_dir()):
        info = _read_info(compound_dir / "compound.info")
        spectrum_id = info["name"]
        if spectrum_id not in expected:
            raise ValueError(f"Unexpected SIRIUS spectrum ID {spectrum_id}")
        if spectrum_id in observed_ids:
            raise ValueError(f"Duplicate SIRIUS spectrum ID {spectrum_id}")
        observed_ids.add(spectrum_id)

        row = expected[spectrum_id]
        expected_formula = row["formula"]
        expected_adduct = row["ionization"].replace(" ", "")
        expected_tree_formula, normalization = _expected_tree_formula(
            expected_formula, expected_adduct
        )
        ms_headers = _read_ms_headers(compound_dir / "spectrum.ms")
        tree = _read_single_tree(compound_dir / "trees")
        observed_tree_formula = tree["molecularFormula"]
        observed_tree_adduct = tree["annotations"]["precursorIonType"].replace(
            " ", ""
        )
        observed_compound_adduct = info["ionType"].replace(" ", "")
        observed_input_formula = ms_headers.get("formula")
        observed_input_adduct = ms_headers.get("ionization", "").replace(" ", "")
        reasons = []
        if observed_input_formula != expected_formula:
            reasons.append("input_formula")
        if observed_input_adduct != expected_adduct:
            reasons.append("input_adduct")
        if observed_compound_adduct != expected_adduct:
            reasons.append("compound_adduct")
        if observed_tree_formula != expected_tree_formula:
            reasons.append("tree_formula")
        if observed_tree_adduct != expected_adduct:
            reasons.append("tree_adduct")
        record = {
            "spec": spectrum_id,
            "candidate_rank": int(row.get("candidate_rank", "1")),
            "expected_formula": expected_formula,
            "expected_tree_formula": expected_tree_formula,
            "expected_adduct": expected_adduct,
            "observed_input_formula": observed_input_formula,
            "observed_input_adduct": observed_input_adduct,
            "observed_compound_adduct": observed_compound_adduct,
            "observed_tree_formula": observed_tree_formula,
            "observed_tree_adduct": observed_tree_adduct,
            "formula_normalization": normalization,
            "reasons": reasons,
        }
        evidence.append(record)
        if reasons:
            mismatches.append(record)

    if observed_ids != set(expected):
        raise ValueError(
            f"SIRIUS/labels ID mismatch: missing={sorted(set(expected)-observed_ids)[:5]}, "
            f"extra={sorted(observed_ids-set(expected))[:5]}"
        )
    evidence_digest = hashlib.sha256(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    report = {
        "schema_version": 1,
        "kind": "SIRIUS formula/adduct consistency audit",
        "rows": len(evidence),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "minimum_candidate_ranks": {
            row["spec"]: row["candidate_rank"] + int(bool(row["reasons"]))
            for row in evidence
        },
        "labels_sha256": sha256_file(labels_path),
        "sirius_tree_evidence_sha256": evidence_digest,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if manifest_path is not None and not mismatches:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("labels_sha256") != report["labels_sha256"]:
            raise ValueError("Formula manifest is not bound to the audited labels")
        manifest.update(
            {
                "sirius_consistency_validated": True,
                "sirius_consistency_mismatch_count": 0,
                "sirius_consistency_audit_sha256": sha256_file(output_path),
                "sirius_tree_evidence_sha256": evidence_digest,
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_project(
        args.project_dir, args.labels, args.output, args.manifest
    )
    print(json.dumps({key: value for key, value in report.items() if key != "mismatches"}))


if __name__ == "__main__":
    main()
