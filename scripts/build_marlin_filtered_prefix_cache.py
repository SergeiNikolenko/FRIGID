#!/usr/bin/env python3
# ruff: noqa: E402
"""Materialize the eligible prefix used to warm MARLIN's shuffle buffer."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import datasets
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marlin.dataset import verify_snapshot_manifest
from marlin.tokenizer import load_safe_tokenizer
from marlin.training import MarlinTrainingFilter
from marlin.warm_start import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--file-list-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--exclude-inchikeys", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--rows", type=int, default=100000)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows <= 0:
        raise ValueError("--rows must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"cache output already exists: {args.output_dir}")
    shards, source_manifest_sha256 = verify_snapshot_manifest(
        args.snapshot_manifest,
        expected_dataset=args.dataset,
        expected_revision=args.revision,
        expected_file_list_sha256=args.file_list_sha256,
        verify_hashes=False,
    )
    tokenizer = load_safe_tokenizer(args.tokenizer)
    eligible = MarlinTrainingFilter(
        tokenizer,
        args.max_length,
        args.exclude_inchikeys,
    )
    stream = datasets.load_dataset(
        "parquet", data_files={"train": shards}, split="train", streaming=True
    )
    safes: list[str] = []
    raw_rows_consumed = 0
    for raw_rows_consumed, row in enumerate(stream, start=1):
        if eligible(row):
            safes.append(str(row.get("safe", row.get("input"))))
            if len(safes) == args.rows:
                break
    if len(safes) != args.rows:
        raise RuntimeError(f"source ended after {len(safes)} eligible rows")

    temporary = args.output_dir.with_name(args.output_dir.name + ".tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    shard = temporary / "eligible-prefix.parquet"
    pq.write_table(
        pa.table({"safe": safes}), shard, compression="zstd", compression_level=6
    )
    manifest = {
        "schema_version": 1,
        "kind": "MARLIN filtered SAFE prefix cache",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.snapshot_manifest),
        "source_manifest_sha256": source_manifest_sha256,
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "exclusion_sha256": sha256_file(args.exclude_inchikeys),
        "max_length": args.max_length,
        "eligible_rows": len(safes),
        "raw_rows_consumed": raw_rows_consumed,
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
