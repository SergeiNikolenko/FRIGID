#!/usr/bin/env python
"""Build deterministic molecule-disjoint calibration and evaluation partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from frigid.encoder_benchmark import (  # noqa: E402
    deterministic_cluster_partitions,
    sha256_file,
    structure_identifiers,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Partition reference spectra by deterministic InChIKey clusters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--id-column", default="spec_name")
    parser.add_argument("--index-column", default="fingerprint_index")
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    metadata = pd.read_csv(args.metadata)
    required = [args.index_column, args.id_column]
    missing = [column for column in required if column not in metadata]
    if missing:
        raise ValueError(f"Reference metadata is missing required columns: {missing}")
    if metadata.empty:
        raise ValueError(f"Reference metadata is empty: {args.metadata}")

    numeric_index = pd.to_numeric(metadata[args.index_column], errors="raise").to_numpy()
    if not np.equal(numeric_index, numeric_index.astype(np.int64)).all():
        raise ValueError(f"Column {args.index_column!r} must contain integer indexes")
    metadata = metadata.assign(
        **{args.index_column: numeric_index.astype(np.int64)}
    ).sort_values(args.index_column, kind="stable")
    expected_index = np.arange(len(metadata), dtype=np.int64)
    if not np.array_equal(metadata[args.index_column].to_numpy(), expected_index):
        raise ValueError(
            f"Column {args.index_column!r} must contain every index from 0 to "
            f"{len(metadata) - 1} exactly once"
        )
    spectrum_ids = metadata[args.id_column].astype(str).str.strip()
    if (
        spectrum_ids.eq("").any()
        or spectrum_ids.str.lower().eq("nan").any()
        or spectrum_ids.duplicated().any()
    ):
        raise ValueError(f"Column {args.id_column!r} must contain unique non-empty IDs")
    metadata[args.id_column] = spectrum_ids

    metadata["benchmark_partition"] = deterministic_cluster_partitions(
        metadata,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    cluster_ids = structure_identifiers(metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "encoder_benchmark_partitions.csv"
    metadata.to_csv(manifest_path, index=False)

    partition_rows = metadata["benchmark_partition"].value_counts().sort_index()
    cluster_frame = pd.DataFrame(
        {"cluster_id": cluster_ids, "partition": metadata["benchmark_partition"]}
    ).drop_duplicates()
    partition_clusters = cluster_frame["partition"].value_counts().sort_index()
    summary = {
        "source_metadata": str(Path(args.metadata).resolve()),
        "source_metadata_sha256": sha256_file(args.metadata),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "ordered_spectrum_ids_sha256": hashlib.sha256(
            "\n".join(spectrum_ids).encode("utf-8")
        ).hexdigest(),
        "seed": args.seed,
        "calibration_fraction": args.calibration_fraction,
        "rows": {name: int(count) for name, count in partition_rows.items()},
        "clusters": {name: int(count) for name, count in partition_clusters.items()},
    }
    (output_dir / "partition_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
