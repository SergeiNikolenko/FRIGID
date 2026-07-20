#!/usr/bin/env python3
"""Audit released MIST-CF split membership for a row-locked evaluation set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def read_rows(path: Path, delimiter: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def audit_split(
    split_path: Path,
    metadata_path: Path,
    *,
    split_id_column: str = "spec",
    fold_column: str = "Fold_0",
    metadata_id_column: str = "spec_name",
) -> dict:
    split_rows = read_rows(split_path, "\t")
    metadata_rows = read_rows(metadata_path, ",")
    split_by_id = {row[split_id_column]: row[fold_column] for row in split_rows}
    if len(split_by_id) != len(split_rows):
        raise ValueError("MIST-CF split contains duplicate spectrum IDs")

    evaluation_ids = [row[metadata_id_column] for row in metadata_rows]
    if len(set(evaluation_ids)) != len(evaluation_ids):
        raise ValueError("evaluation metadata contains duplicate spectrum IDs")
    missing = [spectrum_id for spectrum_id in evaluation_ids if spectrum_id not in split_by_id]
    if missing:
        raise ValueError(f"MIST-CF split is missing {len(missing)} evaluation IDs")

    memberships = [
        {"spec_name": spectrum_id, "fold": split_by_id[spectrum_id]}
        for spectrum_id in evaluation_ids
    ]
    evaluation_counts = Counter(row["fold"] for row in memberships)
    contaminated = sum(evaluation_counts[fold] for fold in ("train", "val"))
    return {
        "official_split_rows": len(split_rows),
        "official_split_counts": dict(Counter(split_by_id.values())),
        "evaluation_rows": len(evaluation_ids),
        "evaluation_split_counts": dict(evaluation_counts),
        "train_or_validation_overlap": contaminated,
        "released_checkpoint_is_evaluation_disjoint": contaminated == 0,
        "memberships": memberships,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = {
        "schema_version": 1,
        "kind": "released MIST-CF split overlap audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "clean_room_reproduction": True,
        "author_code_available_at_start": False,
        "release": {
            "doi": "10.5281/zenodo.8151490",
            "archive_md5": "ac2277361b7d0bf48288e212d2f8dea3",
            "archive_sha256": sha256_file(args.archive),
            "split_sha256": sha256_file(args.split),
            "checkpoint_sha256": sha256_file(args.checkpoint),
        },
        "inputs": {
            "split": str(args.split),
            "metadata": str(args.metadata),
            "checkpoint": str(args.checkpoint),
            "archive": str(args.archive),
        },
        "audit": audit_split(args.split, args.metadata),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
