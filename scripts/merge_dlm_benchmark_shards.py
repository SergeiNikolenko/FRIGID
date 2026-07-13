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
    "fingerprint_drift.csv",
    "paired_comparison.csv",
)
COMPLETE_QUERY_FILES = frozenset(FILES) - {"prediction_scores_mist_binary.csv"}
INPUTS = (
    "config",
    "data_split",
    "labels",
    "mist_checkpoint",
    "dlm_checkpoint",
    "expected_manifest",
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


def manifest_signature(manifest: dict, path: Path) -> dict:
    if manifest.get("schema_version") != 2:
        raise ValueError(f"Unsupported shard manifest schema: {path}")
    if manifest.get("purpose") != "frigid_msg_full_dlm_shard":
        raise ValueError(f"Unexpected shard manifest purpose: {path}")
    if manifest.get("status") != "completed":
        raise ValueError(f"Shard is not completed: {path}")
    if manifest.get("code", {}).get("dirty") is not False:
        raise ValueError(f"Shard code checkout was dirty: {path}")

    inputs = manifest.get("inputs", {})
    input_hashes = {}
    for name in INPUTS:
        sha256 = inputs.get(name, {}).get("sha256")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError(f"Missing {name} SHA-256 in {path}")
        input_hashes[name] = sha256

    settings = manifest.get("settings")
    if not isinstance(settings, dict) or not settings:
        raise ValueError(f"Missing frozen settings in {path}")
    source_name = manifest.get("source_name")
    code_commit = manifest.get("code", {}).get("commit")
    if not source_name or not code_commit:
        raise ValueError(f"Missing source name or code commit in {path}")
    return {
        "source_name": source_name,
        "code_commit": code_commit,
        "input_hashes": input_hashes,
        "settings": settings,
    }


def shard_range(manifest: dict, expected_size: int, path: Path) -> range:
    selection = manifest.get("selection", {})
    if selection.get("split") != "test":
        raise ValueError(f"Shard selection is not the test split: {path}")
    start = selection.get("start_index")
    size = selection.get("max_spectra")
    if not isinstance(start, int) or not isinstance(size, int) or start < 0 or size <= 0:
        raise ValueError(f"Invalid shard range in {path}")
    if start + size > expected_size:
        raise ValueError(f"Shard range exceeds expected manifest in {path}")
    return range(start, start + size)


def aggregate_statistics(details: pd.DataFrame) -> dict:
    numeric = details.apply(pd.to_numeric, errors="ignore")
    total_formula_matched = pd.to_numeric(
        details["total_formula_matched"], errors="raise"
    )
    successful = total_formula_matched > 0
    attempts_per_match = pd.to_numeric(
        details.loc[successful, "total_generated"], errors="raise"
    ) / total_formula_matched.loc[successful]

    def mean(column: str) -> float:
        return float(pd.to_numeric(details[column], errors="raise").mean())

    return {
        "total_spectra": len(details),
        "exact_match_top1": mean("exact_match_top1"),
        "exact_match_top10": mean("exact_match_top10"),
        "tanimoto_top1_mean": mean("tanimoto_top1"),
        "tanimoto_top10_mean": mean("tanimoto_top10"),
        "mist_tanimoto_mean": mean("mist_tanimoto"),
        "avg_formula_matches": mean("total_formula_matched"),
        "avg_predictions_collected": mean("formula_matches_collected"),
        "avg_total_generated": mean("total_generated"),
        "formula_match_success_rate": float(successful.mean()),
        "avg_attempts_to_match": float(attempts_per_match.mean())
        if successful.any()
        else 0.0,
        "never_matched_rate": float((~successful).mean()),
        "total_generation_time_seconds": float(numeric["generation_time"].sum()),
    }


def merge_shards(shard_dirs: list[Path], expected_manifest: Path, output_dir: Path) -> dict:
    if not shard_dirs:
        raise ValueError("At least one shard is required")
    expected = load_manifest(expected_manifest)
    expected_set = set(expected)
    seen: set[str] = set()
    per_file: dict[str, list[pd.DataFrame]] = {name: [] for name in FILES}
    shard_records = []
    expected_manifest_sha256 = sha256_file(expected_manifest)
    frozen_signature: dict | None = None

    for shard_dir in shard_dirs:
        manifest_path = shard_dir / "RUN_MANIFEST.json"
        if not manifest_path.is_file():
            raise ValueError(f"Missing shard manifest: {manifest_path}")
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        signature = manifest_signature(run_manifest, manifest_path)
        if signature["input_hashes"]["expected_manifest"] != expected_manifest_sha256:
            raise ValueError(f"Shard expected-manifest hash mismatch: {manifest_path}")
        if frozen_signature is None:
            frozen_signature = signature
        elif signature != frozen_signature:
            raise ValueError(f"Frozen shard provenance/settings mismatch: {manifest_path}")

        selection_range = shard_range(run_manifest, len(expected), manifest_path)
        expected_shard_queries = set(expected[selection_range.start : selection_range.stop])
        frames: dict[str, pd.DataFrame] = {}
        shard_queries: set[str] | None = None
        output_hashes = run_manifest.get("outputs", {})
        for filename in FILES:
            path = shard_dir / filename
            if not path.is_file():
                raise ValueError(f"Missing {filename}: {path}")
            recorded_sha256 = output_hashes.get(filename, {}).get("sha256")
            actual_sha256 = sha256_file(path)
            if recorded_sha256 != actual_sha256:
                raise ValueError(f"Shard output SHA-256 mismatch: {path}")
            frame = pd.read_csv(path, dtype=str).fillna("")
            column = query_column(frame, path)
            queries = set(frame[column].astype(str))
            if not queries.issubset(expected_set):
                raise ValueError(f"Shard queries are outside expected manifest: {path}")
            if filename in COMPLETE_QUERY_FILES:
                if queries != expected_shard_queries:
                    raise ValueError(f"Shard file does not match declared range: {path}")
                if frame[column].duplicated().any():
                    raise ValueError(f"Shard file repeats query rows: {path}")
                if shard_queries is None:
                    shard_queries = queries
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
            "start_index": selection_range.start,
            "max_spectra": len(selection_range),
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

    aggregate_path = output_dir / "aggregate_statistics.json"
    aggregate = aggregate_statistics(pd.read_csv(output_dir / "detailed_results.csv"))
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    outputs[aggregate_path.name] = {
        "path": str(aggregate_path.resolve()),
        "sha256": sha256_file(aggregate_path),
        "row_count": 1,
    }

    manifest = {
        "schema_version": 2,
        "purpose": "ordered_dlm_benchmark_shard_merge",
        "frozen_signature": frozen_signature,
        "expected_manifest": {
            "path": str(expected_manifest.resolve()),
            "sha256": expected_manifest_sha256,
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
