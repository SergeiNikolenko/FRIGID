#!/usr/bin/env python
"""Join locked benchmark rows with categorical MSG error-analysis fields."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from frigid.encoder_benchmark import sha256_file  # noqa: E402


ANALYSIS_COLUMNS = ("dataset", "ionization", "formula", "instrument")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich row-locked encoder metadata for stratified error analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reference-metadata", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--reference-id-column", default="spec_name")
    parser.add_argument("--labels-id-column", default="spec")
    return parser.parse_args()


def enrich_metadata(
    reference: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    reference_id_column: str,
    labels_id_column: str,
) -> pd.DataFrame:
    """Add selected label fields without changing benchmark row order."""

    if reference_id_column not in reference:
        raise ValueError(f"Reference metadata is missing {reference_id_column!r}")
    if labels_id_column not in labels:
        raise ValueError(f"Labels are missing {labels_id_column!r}")
    if reference[reference_id_column].duplicated().any():
        raise ValueError("Reference metadata contains duplicate spectrum IDs")
    if labels[labels_id_column].duplicated().any():
        raise ValueError("Labels contain duplicate spectrum IDs")
    available_columns = [column for column in ANALYSIS_COLUMNS if column in labels]
    if not available_columns:
        raise ValueError(
            f"Labels contain none of the error-analysis columns {ANALYSIS_COLUMNS}"
        )
    conflicts = sorted(set(available_columns) & set(reference.columns))
    if conflicts:
        raise ValueError(f"Reference metadata already contains label columns: {conflicts}")

    label_fields = labels[[labels_id_column, *available_columns]].copy()
    enriched = reference.merge(
        label_fields,
        how="left",
        left_on=reference_id_column,
        right_on=labels_id_column,
        validate="one_to_one",
        sort=False,
    )
    if labels_id_column != reference_id_column:
        enriched = enriched.drop(columns=[labels_id_column])
    if enriched[available_columns].isna().any(axis=None):
        missing = enriched.loc[
            enriched[available_columns].isna().any(axis=1), reference_id_column
        ].head(5)
        raise ValueError(f"Reference IDs missing label fields: {missing.tolist()}")
    if not enriched[reference_id_column].equals(
        reference[reference_id_column].reset_index(drop=True)
    ):
        raise ValueError("Metadata enrichment changed benchmark row order")
    return enriched


def main() -> int:
    args = parse_args()
    reference_path = Path(args.reference_metadata).resolve()
    labels_path = Path(args.labels).resolve()
    output_path = Path(args.output).resolve()
    summary_path = Path(args.summary or output_path.with_suffix(".summary.json")).resolve()
    for path in (output_path, summary_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    reference = pd.read_csv(reference_path, dtype=str)
    labels = pd.read_csv(labels_path, sep="\t", dtype=str)
    enriched = enrich_metadata(
        reference,
        labels,
        reference_id_column=args.reference_id_column,
        labels_id_column=args.labels_id_column,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False)
    summary = {
        "rows": len(enriched),
        "reference_metadata": str(reference_path),
        "reference_metadata_sha256": sha256_file(reference_path),
        "labels": str(labels_path),
        "labels_sha256": sha256_file(labels_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "analysis_columns": [
            column for column in ANALYSIS_COLUMNS if column in enriched.columns
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
