#!/usr/bin/env python
"""Merge ordered MCES shards and compare the frozen union with control."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_PATH = PROJECT_ROOT / "scripts"
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

from compare_paired_benchmark_runs import (  # noqa: E402
    cluster_bootstrap_mean_interval,
    hash_names,
)
from evaluate_mces_predictions import MCES_COLUMNS, sha256_file  # noqa: E402


REFERENCE_VARIANT = "control"
CANDIDATE_VARIANT = "union"


def load_expected_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    required = {"spec_name", "inchikey_first_block"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Expected manifest is missing columns {missing}: {path}")
    names = frame["spec_name"].astype(str).tolist()
    clusters = frame["inchikey_first_block"].astype(str).tolist()
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("Expected manifest spec_name values must be non-empty and unique")
    if any(not cluster for cluster in clusters):
        raise ValueError("Expected manifest contains an empty molecule cluster")
    return frame.reset_index(drop=True)


def manifest_signature(manifest: dict[str, Any], path: Path) -> dict[str, Any]:
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported MCES shard manifest schema: {path}")
    if manifest.get("purpose") != "frigid_full_mces_shard":
        raise ValueError(f"Unexpected MCES shard purpose: {path}")
    if manifest.get("status") != "completed" or manifest.get("exit_code") != 0:
        raise ValueError(f"MCES shard is not completed successfully: {path}")
    code = manifest.get("code", {})
    if not code.get("commit") or code.get("dirty") is not False:
        raise ValueError(f"MCES shard code provenance is invalid: {path}")

    inputs = manifest.get("inputs", {})
    expected_manifest = inputs.get("expected_manifest", {})
    runtime_manifest = inputs.get("runtime_manifest", {})
    predictions = inputs.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError(f"MCES shard predictions are missing: {path}")
    for label, record in (
        ("expected manifest", expected_manifest),
        ("runtime manifest", runtime_manifest),
    ):
        if not record.get("path") or not _is_sha256(record.get("sha256")):
            raise ValueError(f"MCES shard {label} provenance is invalid: {path}")
    for prediction in predictions:
        if (
            not prediction.get("name")
            or not prediction.get("path")
            or not _is_sha256(prediction.get("sha256"))
        ):
            raise ValueError(f"MCES shard prediction provenance is invalid: {path}")

    settings = manifest.get("settings")
    runtime = manifest.get("runtime")
    if not isinstance(settings, dict) or not settings:
        raise ValueError(f"MCES shard settings are missing: {path}")
    if not isinstance(runtime, dict) or not runtime:
        raise ValueError(f"MCES shard runtime provenance is missing: {path}")
    variant_order = settings.get("variant_order")
    if variant_order != [REFERENCE_VARIANT, CANDIDATE_VARIANT]:
        raise ValueError(f"Unexpected frozen MCES variant order in {path}: {variant_order}")
    return {
        "code_commit": code["commit"],
        "inputs": inputs,
        "runtime": runtime,
        "settings": settings,
    }


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def shard_bounds(manifest: dict[str, Any], expected_size: int, path: Path) -> tuple[int, int]:
    selection = manifest.get("selection", {})
    start = selection.get("start_index")
    size = selection.get("max_spectra")
    total = selection.get("expected_total")
    if total != expected_size:
        raise ValueError(f"MCES shard expected-total mismatch: {path}")
    if not isinstance(start, int) or not isinstance(size, int) or start < 0 or size <= 0:
        raise ValueError(f"Invalid MCES shard range: {path}")
    stop = start + size
    if stop > expected_size:
        raise ValueError(f"MCES shard range exceeds the expected manifest: {path}")
    progress = manifest.get("progress", {})
    if progress.get("processed_spectra") != size or progress.get("expected_spectra") != size:
        raise ValueError(f"MCES shard progress is incomplete: {path}")
    return start, stop


def load_shard_rows(
    shard_dir: Path,
    manifest: dict[str, Any],
    expected_names: list[str],
    variant_order: list[str],
) -> pd.DataFrame:
    path = shard_dir / "per_sample_mces.csv"
    if not path.is_file():
        raise ValueError(f"Missing MCES shard output: {path}")
    output = manifest.get("outputs", {}).get("per_sample_mces", {})
    if Path(str(output.get("path", ""))).expanduser().resolve() != path.resolve():
        raise ValueError(f"MCES shard output path mismatch: {path}")
    if output.get("sha256") != sha256_file(path):
        raise ValueError(f"MCES shard output SHA-256 mismatch: {path}")

    frame = pd.read_csv(path, dtype={"spec_name": str, "variant": str})
    required = {"spec_name", "variant", *MCES_COLUMNS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"MCES shard output is missing columns {missing}: {path}")
    if output.get("row_count") != len(frame):
        raise ValueError(f"MCES shard output row-count mismatch: {path}")
    expected_pairs = [
        (spec_name, variant)
        for spec_name in expected_names
        for variant in variant_order
    ]
    observed_pairs = list(zip(frame["spec_name"].astype(str), frame["variant"].astype(str)))
    if observed_pairs != expected_pairs:
        raise ValueError(f"MCES shard rows do not match the declared ordered range: {path}")
    values = frame.loc[:, MCES_COLUMNS].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"MCES shard contains non-finite metric values: {path}")
    frame.loc[:, MCES_COLUMNS] = values
    return frame.loc[:, ["spec_name", "variant", *MCES_COLUMNS]]


def aggregate_statistics(frame: pd.DataFrame, variant_order: list[str]) -> dict[str, Any]:
    statistics: dict[str, Any] = {"query_count": int(frame["spec_name"].nunique()), "variants": {}}
    for variant in variant_order:
        selected = frame.loc[frame["variant"] == variant]
        statistics["variants"][variant] = {
            column: {
                "mean": float(pd.to_numeric(selected[column], errors="raise").mean()),
                "median": float(pd.to_numeric(selected[column], errors="raise").median()),
            }
            for column in MCES_COLUMNS
        }
    return statistics


def paired_comparison(
    frame: pd.DataFrame,
    expected_manifest: pd.DataFrame,
    *,
    bootstrap_resamples: int,
    confidence: float,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    names = expected_manifest["spec_name"].astype(str).tolist()
    clusters = expected_manifest["inchikey_first_block"].astype(str).to_numpy()
    reference = frame.loc[frame["variant"] == REFERENCE_VARIANT].set_index("spec_name")
    candidate = frame.loc[frame["variant"] == CANDIDATE_VARIANT].set_index("spec_name")
    if reference.index.tolist() != names or candidate.index.tolist() != names:
        raise ValueError("MCES variants do not match the locked manifest order")

    paired = pd.DataFrame(
        {"spec_name": names, "bootstrap_cluster": clusters}
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "reference": REFERENCE_VARIANT,
        "candidate": CANDIDATE_VARIANT,
        "lower_is_better": True,
        "delta_definition": "candidate_minus_reference",
        "n_pairs": len(names),
        "subset_sha256_ordered": hash_names(names),
        "subset_sha256_sorted": hash_names(sorted(names)),
        "bootstrap": {
            "resamples": bootstrap_resamples,
            "confidence": confidence,
            "seed": seed,
            "unit": "molecule",
            "cluster_column": "inchikey_first_block",
            "n_clusters": int(np.unique(clusters).size),
        },
        "metrics": {},
    }
    rng = np.random.default_rng(seed)
    for column in MCES_COLUMNS:
        reference_values = reference.loc[names, column].to_numpy(dtype=np.float64)
        candidate_values = candidate.loc[names, column].to_numpy(dtype=np.float64)
        deltas = candidate_values - reference_values
        ci_low, ci_high = cluster_bootstrap_mean_interval(
            deltas, clusters, bootstrap_resamples, confidence, rng
        )
        paired[f"{column}_reference"] = reference_values
        paired[f"{column}_candidate"] = candidate_values
        paired[f"{column}_delta"] = deltas
        summary["metrics"][column] = {
            "reference_mean": float(reference_values.mean()),
            "candidate_mean": float(candidate_values.mean()),
            "mean_delta": float(deltas.mean()),
            "median_delta": float(np.median(deltas)),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "improved": int((deltas < 0).sum()),
            "regressed": int((deltas > 0).sum()),
            "ties": int((deltas == 0).sum()),
        }
    return summary, paired


def write_comparison(output_dir: Path, summary: dict[str, Any], paired: pd.DataFrame) -> dict[str, Any]:
    output_dir.mkdir()
    summary_path = output_dir / "comparison_summary.json"
    paired_path = output_dir / "paired_deltas.csv"
    bootstrap_path = output_dir / "bootstrap_ci.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paired.to_csv(paired_path, index=False)
    bootstrap = {
        "n_pairs": summary["n_pairs"],
        "bootstrap": summary["bootstrap"],
        "lower_is_better": True,
        "metrics": {
            name: {
                "mean_delta": values["mean_delta"],
                "ci_low": values["ci_low"],
                "ci_high": values["ci_high"],
            }
            for name, values in summary["metrics"].items()
        },
    }
    bootstrap_path.write_text(json.dumps(bootstrap, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        str(path.relative_to(output_dir.parent)): {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "row_count": len(paired) if path == paired_path else 1,
        }
        for path in (summary_path, paired_path, bootstrap_path)
    }


def merge_shards(
    shard_dirs: list[Path],
    expected_manifest_path: Path,
    output_dir: Path,
    *,
    bootstrap_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
    shard_list_path: Path | None = None,
) -> dict[str, Any]:
    if not shard_dirs:
        raise ValueError("At least one MCES shard is required")
    if output_dir.exists():
        raise ValueError(f"MCES merge output directory already exists: {output_dir}")
    expected_manifest = load_expected_manifest(expected_manifest_path)
    expected_names = expected_manifest["spec_name"].astype(str).tolist()
    expected_sha256 = sha256_file(expected_manifest_path)
    covered = np.zeros(len(expected_names), dtype=bool)
    frozen_signature: dict[str, Any] | None = None
    shard_records: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []

    for raw_shard_dir in shard_dirs:
        shard_dir = raw_shard_dir.expanduser().resolve()
        manifest_path = shard_dir / "RUN_MANIFEST.json"
        if not manifest_path.is_file():
            raise ValueError(f"Missing MCES shard manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        signature = manifest_signature(manifest, manifest_path)
        if signature["inputs"]["expected_manifest"]["sha256"] != expected_sha256:
            raise ValueError(f"MCES shard expected-manifest SHA-256 mismatch: {manifest_path}")
        if frozen_signature is None:
            frozen_signature = signature
        elif signature != frozen_signature:
            raise ValueError(f"Frozen MCES shard provenance/settings mismatch: {manifest_path}")
        start, stop = shard_bounds(manifest, len(expected_names), manifest_path)
        if covered[start:stop].any():
            raise ValueError(f"Overlapping MCES shard range: {manifest_path}")
        shard_frame = load_shard_rows(
            shard_dir,
            manifest,
            expected_names[start:stop],
            signature["settings"]["variant_order"],
        )
        covered[start:stop] = True
        frames.append(shard_frame)
        shard_records.append(
            {
                "path": str(shard_dir),
                "run_manifest_sha256": sha256_file(manifest_path),
                "start_index": start,
                "max_spectra": stop - start,
                "row_count": len(shard_frame),
            }
        )

    missing_indices = np.flatnonzero(~covered).tolist()
    if missing_indices:
        raise ValueError(f"MCES shard coverage is incomplete; missing indices: {missing_indices[:5]}")
    assert frozen_signature is not None
    variant_order = frozen_signature["settings"]["variant_order"]
    merged = pd.concat(frames, ignore_index=True)
    name_order = {name: index for index, name in enumerate(expected_names)}
    variant_rank = {name: index for index, name in enumerate(variant_order)}
    merged["_name_order"] = merged["spec_name"].map(name_order)
    merged["_variant_order"] = merged["variant"].map(variant_rank)
    merged = merged.sort_values(["_name_order", "_variant_order"], kind="stable").drop(
        columns=["_name_order", "_variant_order"]
    )
    expected_pairs = [
        (spec_name, variant)
        for spec_name in expected_names
        for variant in variant_order
    ]
    if list(zip(merged["spec_name"], merged["variant"])) != expected_pairs:
        raise ValueError("Merged MCES rows do not have exact locked coverage")

    aggregate = aggregate_statistics(merged, variant_order)
    comparison, paired = paired_comparison(
        merged,
        expected_manifest,
        bootstrap_resamples=bootstrap_resamples,
        confidence=confidence,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    merged_path = output_dir / "per_sample_mces.csv"
    aggregate_path = output_dir / "aggregate_statistics.json"
    merged.to_csv(merged_path, index=False)
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs = {
        merged_path.name: {
            "path": str(merged_path.resolve()),
            "sha256": sha256_file(merged_path),
            "row_count": len(merged),
        },
        aggregate_path.name: {
            "path": str(aggregate_path.resolve()),
            "sha256": sha256_file(aggregate_path),
            "row_count": 1,
        },
    }
    outputs.update(write_comparison(output_dir / "paired_control_vs_union", comparison, paired))
    merge_manifest: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "ordered_full_mces_shard_merge",
        "status": "completed",
        "frozen_signature": frozen_signature,
        "expected_manifest": {
            "path": str(expected_manifest_path.resolve()),
            "sha256": expected_sha256,
            "query_count": len(expected_names),
        },
        "shard_list": None
        if shard_list_path is None
        else {
            "path": str(shard_list_path.resolve()),
            "sha256": sha256_file(shard_list_path),
        },
        "shards": sorted(shard_records, key=lambda record: record["start_index"]),
        "coverage": {
            "expected_queries": len(expected_names),
            "observed_queries": len(expected_names),
            "observed_rows": len(merged),
            "missing_indices": [],
        },
        "comparison": {
            "reference": REFERENCE_VARIANT,
            "candidate": CANDIDATE_VARIANT,
            "delta_definition": "candidate_minus_reference",
            "lower_is_better": True,
            "bootstrap_resamples": bootstrap_resamples,
            "confidence": confidence,
            "seed": seed,
        },
        "outputs": outputs,
    }
    (output_dir / "MERGE_MANIFEST.json").write_text(
        json.dumps(merge_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return merge_manifest


def load_shard_list(path: Path) -> list[Path]:
    entries = [
        Path(line.strip()).expanduser()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not entries:
        raise ValueError(f"MCES shard list is empty: {path}")
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", action="append", default=[], type=Path)
    parser.add_argument("--shard-list", type=Path)
    parser.add_argument("--expected-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    shard_dirs = list(args.shard)
    shard_list_path = None
    if args.shard_list is not None:
        shard_list_path = args.shard_list.expanduser().resolve()
        shard_dirs.extend(load_shard_list(shard_list_path))
    merge_shards(
        shard_dirs,
        args.expected_manifest.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        bootstrap_resamples=args.bootstrap_resamples,
        confidence=args.confidence,
        seed=args.seed,
        shard_list_path=shard_list_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
