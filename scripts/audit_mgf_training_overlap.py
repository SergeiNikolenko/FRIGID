#!/usr/bin/env python
"""Audit structure overlap between published MGF training data and an evaluation set."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inchikey_first_block(value: str) -> str:
    """Normalize a full or first-block InChIKey to its 14-character structure block."""

    normalized = value.strip().upper().split("-", maxsplit=1)[0]
    if len(normalized) != 14 or not normalized.isalpha():
        raise ValueError(f"Invalid InChIKey or first block: {value!r}")
    return normalized


def iter_mgf_inchikeys(path: str | Path) -> Iterable[str | None]:
    """Yield one normalized InChIKey block per MGF ion block."""

    path = Path(path)
    inside_block = False
    headers: dict[str, str] = {}
    with path.open(errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            upper = line.upper()
            if upper == "BEGIN IONS":
                if inside_block:
                    raise ValueError(f"Nested BEGIN IONS in {path}")
                inside_block = True
                headers = {}
                continue
            if upper == "END IONS":
                if not inside_block:
                    raise ValueError(f"END IONS without BEGIN IONS in {path}")
                value = headers.get("INCHIKEY") or headers.get("INCHI_AUX")
                yield inchikey_first_block(value) if value else None
                inside_block = False
                headers = {}
                continue
            if inside_block and "=" in line:
                key, value = line.split("=", maxsplit=1)
                key = key.strip().upper()
                if key in {"INCHIKEY", "INCHI_AUX"} and key not in headers:
                    headers[key] = value.strip()
    if inside_block:
        raise ValueError(f"Unterminated BEGIN IONS block in {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare published MGF training structures with evaluation metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--training-mgf", action="append", required=True)
    parser.add_argument("--evaluation-metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--id-column", default="spectrum_id")
    parser.add_argument("--structure-column", default="inchi_key_first_block")
    parser.add_argument("--filter-column")
    parser.add_argument("--filter-value")
    parser.add_argument("--deduplicate-column")
    parser.add_argument("--max-examples", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_examples < 0:
        raise ValueError("--max-examples cannot be negative")

    metadata_path = Path(args.evaluation_metadata).resolve()
    metadata = pd.read_csv(metadata_path)
    required = {args.id_column, args.structure_column}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Evaluation metadata is missing columns: {missing}")
    if args.filter_column:
        if args.filter_column not in metadata:
            raise ValueError(
                f"Evaluation metadata is missing filter column {args.filter_column!r}"
            )
        metadata = metadata.loc[
            metadata[args.filter_column].astype(str) == str(args.filter_value)
        ].copy()
    if args.deduplicate_column:
        if args.deduplicate_column not in metadata:
            raise ValueError(
                f"Evaluation metadata is missing deduplication column "
                f"{args.deduplicate_column!r}"
            )
        metadata = metadata.drop_duplicates(args.deduplicate_column, keep="first")
    if metadata.empty:
        raise ValueError("Evaluation metadata selection is empty")

    evaluation_ids = metadata[args.id_column].astype(str).str.strip()
    if evaluation_ids.eq("").any() or evaluation_ids.str.lower().eq("nan").any():
        raise ValueError(f"Evaluation ID column {args.id_column!r} contains empty values")
    evaluation_structures = metadata[args.structure_column].map(
        lambda value: inchikey_first_block(str(value))
    )

    training_counter: Counter[str] = Counter()
    file_summaries = []
    missing_training_structures = 0
    training_rows = 0
    training_paths = [Path(value).resolve() for value in args.training_mgf]
    for path in training_paths:
        file_counter: Counter[str] = Counter()
        file_missing = 0
        file_rows = 0
        for structure in iter_mgf_inchikeys(path):
            file_rows += 1
            if structure is None:
                file_missing += 1
            else:
                file_counter[structure] += 1
                training_counter[structure] += 1
        training_rows += file_rows
        missing_training_structures += file_missing
        file_summaries.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": file_rows,
                "rows_without_inchikey": file_missing,
                "unique_structure_blocks": len(file_counter),
            }
        )

    overlap_mask = evaluation_structures.isin(training_counter)
    overlapping_rows = metadata.loc[overlap_mask].copy()
    overlapping_rows = overlapping_rows.assign(
        normalized_structure=evaluation_structures.loc[overlap_mask].to_numpy(),
        training_spectrum_count=[
            training_counter[value] for value in evaluation_structures.loc[overlap_mask]
        ],
    )
    overlapping_structures = set(evaluation_structures.loc[overlap_mask])
    examples = overlapping_rows[
        [args.id_column, args.structure_column, "normalized_structure", "training_spectrum_count"]
    ].head(args.max_examples)

    result = {
        "schema_version": 1,
        "training": {
            "files": file_summaries,
            "rows": training_rows,
            "rows_without_inchikey": missing_training_structures,
            "unique_structure_blocks": len(training_counter),
        },
        "evaluation": {
            "metadata": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "rows": len(metadata),
            "unique_structure_blocks": int(evaluation_structures.nunique()),
            "id_column": args.id_column,
            "structure_column": args.structure_column,
            "filter_column": args.filter_column,
            "filter_value": args.filter_value,
            "deduplicate_column": args.deduplicate_column,
        },
        "overlap": {
            "rows": int(overlap_mask.sum()),
            "row_fraction": float(overlap_mask.mean()),
            "unique_structure_blocks": len(overlapping_structures),
            "structure_fraction": len(overlapping_structures)
            / int(evaluation_structures.nunique()),
            "examples": examples.to_dict(orient="records"),
        },
    }
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
