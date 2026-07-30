#!/usr/bin/env python3
# ruff: noqa: E402
"""Materialize a finite prefix of MARLIN's canonical shuffled training stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import datasets
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marlin.dataset import (
    verify_filtered_prefix_cache,
    verify_snapshot_manifest,
)
from marlin.tokenizer import load_safe_tokenizer
from marlin.training import MarlinTrainingFilter
from marlin.warm_start import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--filtered-prefix-manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--file-list-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--exclude-inchikeys", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--shuffle-buffer", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--row-group-size", type=int, default=10000)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows <= 0 or args.row_group_size <= 0:
        raise ValueError("--rows and --row-group-size must be positive")
    if args.shuffle_buffer <= 0:
        raise ValueError("--shuffle-buffer must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"cache output already exists: {args.output_dir}")

    shards, source_manifest_sha256 = verify_snapshot_manifest(
        args.snapshot_manifest,
        expected_dataset=args.dataset,
        expected_revision=args.revision,
        expected_file_list_sha256=args.file_list_sha256,
        verify_hashes=False,
    )
    tokenizer_sha256 = sha256_file(args.tokenizer)
    exclusion_sha256 = sha256_file(args.exclude_inchikeys)
    filtered_prefix_manifest_sha256 = sha256_file(
        args.filtered_prefix_manifest
    )
    cached_prefix_shard, raw_rows_consumed = verify_filtered_prefix_cache(
        args.filtered_prefix_manifest,
        source_manifest_sha256=source_manifest_sha256,
        tokenizer_sha256=tokenizer_sha256,
        exclusion_sha256=exclusion_sha256,
        max_length=args.max_length,
        minimum_rows=args.shuffle_buffer,
    )

    tokenizer = load_safe_tokenizer(args.tokenizer)
    training_filter = MarlinTrainingFilter(
        tokenizer,
        args.max_length,
        args.exclude_inchikeys,
    )
    source = datasets.load_dataset(
        "parquet",
        data_files={"train": shards},
        split="train",
        streaming=True,
    )
    cached_prefix = datasets.load_dataset(
        "parquet",
        data_files={"train": [cached_prefix_shard]},
        split="train",
        streaming=True,
    )
    tail = source.skip(raw_rows_consumed).filter(training_filter)
    stream = datasets.concatenate_datasets([cached_prefix, tail]).shuffle(
        seed=args.seed,
        buffer_size=args.shuffle_buffer,
    )

    temporary = args.output_dir.with_name(args.output_dir.name + ".tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    shard = temporary / "shuffled-stream.parquet"
    schema = pa.schema([("safe", pa.string())])
    rows_written = 0
    batch: list[str] = []
    with pq.ParquetWriter(
        shard,
        schema,
        compression="zstd",
        compression_level=6,
    ) as writer:
        for row in stream:
            batch.append(str(row.get("safe", row.get("input"))))
            if len(batch) == args.row_group_size:
                writer.write_table(pa.table({"safe": batch}, schema=schema))
                rows_written += len(batch)
                print(f"cached_rows={rows_written}", flush=True)
                batch.clear()
            if rows_written + len(batch) == args.rows:
                break
        if batch:
            writer.write_table(pa.table({"safe": batch}, schema=schema))
            rows_written += len(batch)
    if rows_written != args.rows:
        raise RuntimeError(
            f"source ended after {rows_written} rows; expected {args.rows}"
        )

    manifest = {
        "schema_version": 1,
        "kind": "MARLIN shuffled eligible stream cache",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.snapshot_manifest),
        "source_manifest_sha256": source_manifest_sha256,
        "filtered_prefix_manifest": str(args.filtered_prefix_manifest),
        "filtered_prefix_manifest_sha256": filtered_prefix_manifest_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "exclusion_sha256": exclusion_sha256,
        "max_length": args.max_length,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "rows": rows_written,
        "stream_contract_sha256": hashlib.sha256(
            json.dumps(
                {
                    "source_manifest_sha256": source_manifest_sha256,
                    "filtered_prefix_manifest_sha256": (
                        filtered_prefix_manifest_sha256
                    ),
                    "tokenizer_sha256": tokenizer_sha256,
                    "exclusion_sha256": exclusion_sha256,
                    "max_length": args.max_length,
                    "seed": args.seed,
                    "shuffle_buffer": args.shuffle_buffer,
                    "rows": rows_written,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "shard": shard.name,
        "shard_size_bytes": shard.stat().st_size,
        "shard_sha256": sha256_file(shard),
    }
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    temporary.rename(args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
