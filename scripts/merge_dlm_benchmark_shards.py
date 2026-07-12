#!/usr/bin/env python
"""Merge exact ordered DLM benchmark shards with coverage checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


FILES = (
    "predictions_mist_binary.csv",
    "prediction_scores_mist_binary.csv",
    "detailed_results.csv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[str]:
    frame = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if "spec_name" not in frame.columns:
        raise ValueError(f"Manifest is missing spec_name: {path}")
    names = frame["spec_name"].astype(str).tolist()
    if not names or len(names) != len(set(names)):
        raise ValueError(f"Manifest spec_name values must be non-empty and unique: {path}")
    return names


def query_column(frame: pd.DataFrame, path: Path) -> str:
    for name in ("spec_name", "name"):
        if name in frame.columns:
            return name
    raise ValueError(f"No query column in {path}")


def merge_shards(shard_dirs: list[Path], expected_manifest: Path, output_dir: Path) -> dict:
    if not shard_dirs:
        raise ValueError("At least one shard is required")
    expected = load_manifest(expected_manifest)
    expected_set = set(expected)
    seen: set[str] = set()
    per_file: dict[str, list[pd.DataFrame]] = {name: [] for name in FILES}
    shard_records = []

    for shard_dir in shard_dirs:
        manifest_path = shard_dir / "RUN_MANIFEST.json"
        if not manifest_path.is_file():
            raise ValueError(f"Missing shard manifest: {manifest_path}")
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frames: dict[str, pd.DataFrame] = {}
        shard_queries: set[str] | None = None
        for filename in FILES:
            path = shard_dir / filename
            if not path.is_file():
                raise ValueError(f"Missing {filename}: {path}")
            frame = pd.read_csv(path, dtype=str).fillna("")
            column = query_column(frame, path)
            queries = set(frame[column].astype(str))
            if not queries or not queries.issubset(expected_set):
                raise ValueError(f"Shard queries are outside expected manifest: {path}")
            if shard_queries is None:
                shard_queries = queries
            elif queries != shard_queries:
                raise ValueError(f"Shard files disagree on query coverage: {shard_dir}")
            frames[filename] = frame
        assert shard_queries is not None
        overlap = sorted(seen.intersection(shard_queries))
        if overlap:
            raise ValueError(f"Overlapping shard queries: {overlap[:5]}")
        seen.update(shard_queries)
        for filename, frame in frames.items():
            per_file[filename].append(frame)
        shard_records.append({
            "path": str(shard_dir.resolve()),
            "run_manifest_sha256": sha256_file(manifest_path),
            "query_count": len(shard_queries),
            "queries": sorted(shard_queries),
        })

    missing = sorted(expected_set.difference(seen))
    extra = sorted(seen.difference(expected_set))
    if missing or extra:
        raise ValueError(f"Shard coverage mismatch: missing={missing[:5]}, extra={extra[:5]}")

    output_dir.mkdir(parents=True, exist_ok=False)
    order = {name: index for index, name in enumerate(expected)}
    outputs = {}
    for filename, frames in per_file.items():
        merged = pd.concat(frames, ignore_index=True)
        column = query_column(merged, output_dir / filename)
        merged["_frigid_order"] = merged[column].map(order)
        if merged["_frigid_order"].isna().any():
            raise ValueError(f"Merged output contains unknown queries: {filename}")
        merged.sort_values("_frigid_order", kind="stable").drop(
            columns=["_frigid_order"]
        ).to_csv(output_dir / filename, index=False)
        outputs[filename] = {
            "path": str((output_dir / filename).resolve()),
            "sha256": sha256_file(output_dir / filename),
            "row_count": len(merged),
        }

    manifest = {
        "schema_version": 1,
        "purpose": "ordered_dlm_benchmark_shard_merge",
        "expected_manifest": {
            "path": str(expected_manifest.resolve()),
            "sha256": sha256_file(expected_manifest),
            "query_count": len(expected),
        },
        "shards": shard_records,
        "coverage": {"expected": len(expected), "observed": len(seen), "missing": missing, "extra": extra},
        "outputs": outputs,
    }
    (output_dir / "MERGE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", action="append", required=True, type=Path)
    parser.add_argument("--expected-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    merge_shards(args.shard, args.expected_manifest, args.output_dir)
