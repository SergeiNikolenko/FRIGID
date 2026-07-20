#!/usr/bin/env python3
"""Create a MIST-CF split with evaluation connectivities excluded from fitting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_clean_split(
    split_path: Path,
    connectivity_audit_path: Path,
    output_path: Path,
    *,
    split_id_column: str = "spec",
    fold_column: str = "Fold_0",
) -> dict:
    audit_manifest = json.loads(connectivity_audit_path.read_text())
    connectivity_audit = audit_manifest["connectivity_audit"]
    overlap_rows = connectivity_audit["memberships"]
    overlap_ids = {row["spec_name"] for row in overlap_rows}
    if len(overlap_ids) != len(overlap_rows):
        raise ValueError("connectivity audit contains duplicate spectrum IDs")

    with split_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or split_id_column not in fieldnames or fold_column not in fieldnames:
        raise ValueError("split is missing required columns")
    split_ids = {row[split_id_column] for row in rows}
    missing = overlap_ids - split_ids
    if missing:
        raise ValueError(f"split is missing {len(missing)} audited spectrum IDs")

    original_counts = Counter(row[fold_column] for row in rows)
    moved_counts = Counter()
    for row in rows:
        if row[split_id_column] in overlap_ids and row[fold_column] != "test":
            moved_counts[row[fold_column]] += 1
            row[fold_column] = "test"
    clean_counts = Counter(row[fold_column] for row in rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    return {
        "original_split_counts": dict(original_counts),
        "clean_split_counts": dict(clean_counts),
        "connectivity_overlap_spectra": len(overlap_ids),
        "moved_to_test_counts": dict(moved_counts),
        "remaining_train_or_validation_connectivity_overlap": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--connectivity-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    result = make_clean_split(args.split, args.connectivity_audit, args.output)
    manifest = {
        "schema_version": 1,
        "kind": "MIST-CF connectivity-clean split",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": (
            "Force every official-archive spectrum sharing an NPLIB1 evaluation "
            "InChIKey connectivity block into the test fold"
        ),
        "inputs": {
            "split": str(args.split),
            "split_sha256": sha256_file(args.split),
            "connectivity_audit": str(args.connectivity_audit),
            "connectivity_audit_sha256": sha256_file(args.connectivity_audit),
        },
        "output": {
            "split": str(args.output),
            "split_sha256": sha256_file(args.output),
        },
        "result": result,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
