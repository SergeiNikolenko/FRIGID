#!/usr/bin/env python
"""Convert MolForge JSONL predictions to the FRIGID candidate-source schema."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest_spec_names(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "spec_name" not in reader.fieldnames:
            raise ValueError(f"Manifest is missing spec_name: {path}")
        names = [str(row["spec_name"]).strip() for row in reader]
    if not names or any(not name for name in names):
        raise ValueError(f"Manifest contains no usable spec names: {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"Manifest contains duplicate spec names: {path}")
    return names


def load_prediction_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {path}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected an object on line {line_number}: {path}")
            records.append(payload)
    if not records:
        raise ValueError(f"Prediction JSONL is empty: {path}")
    return records


def convert_predictions(
    input_jsonl: Path,
    output_csv: Path,
    spec_manifest: Path,
) -> tuple[int, int]:
    records = load_prediction_records(input_jsonl)
    expected_names = load_manifest_spec_names(spec_manifest)

    actual_names = [str(record.get("spec_name", "")).strip() for record in records]
    if actual_names != expected_names:
        missing = sorted(set(expected_names) - set(actual_names))[:5]
        extra = sorted(set(actual_names) - set(expected_names))[:5]
        raise ValueError(
            "MolForge predictions do not match manifest order: "
            f"expected={len(expected_names)}, actual={len(actual_names)}, "
            f"missing={missing}, extra={extra}"
        )

    rows: list[dict[str, Any]] = []
    queries_with_no_candidates = 0
    for record in records:
        spec_name = str(record["spec_name"]).strip()
        raw_predictions = record.get("pred_smiles_top10")
        if not isinstance(raw_predictions, list):
            raise ValueError(f"pred_smiles_top10 is not a list for {spec_name}")
        predictions = [str(value).strip() for value in raw_predictions if str(value).strip()]
        if not predictions:
            queries_with_no_candidates += 1
            continue
        if len(predictions) > 10:
            raise ValueError(f"MolForge produced more than 10 candidates for {spec_name}")
        threshold = record.get("threshold", "")
        for rank, smiles in enumerate(predictions, start=1):
            rows.append(
                {
                    "query_spec_name": spec_name,
                    "rank": rank,
                    "candidate_smiles": smiles,
                    "source_threshold": threshold,
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "query_spec_name",
                "rank",
                "candidate_smiles",
                "source_threshold",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = output_csv.with_suffix(output_csv.suffix + ".manifest.json")
    manifest_payload = {
        "schema_version": 1,
        "converter": str(Path(__file__).resolve()),
        "input_jsonl": {
            "path": str(input_jsonl),
            "sha256": sha256_file(input_jsonl),
        },
        "spec_manifest": {
            "path": str(spec_manifest),
            "sha256": sha256_file(spec_manifest),
        },
        "output_csv": {
            "path": str(output_csv),
            "sha256": sha256_file(output_csv),
        },
        "query_count": len(records),
        "candidate_count": len(rows),
        "queries_with_no_candidates": queries_with_no_candidates,
        "target_fields_used": [],
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n")
    return len(records), len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--spec-manifest", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queries, candidates = convert_predictions(
        input_jsonl=Path(args.input_jsonl).expanduser().resolve(),
        output_csv=Path(args.output_csv).expanduser().resolve(),
        spec_manifest=Path(args.spec_manifest).expanduser().resolve(),
    )
    print(f"Converted {candidates} candidates for {queries} queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
