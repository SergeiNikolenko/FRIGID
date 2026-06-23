#!/usr/bin/env python3
"""Summarize FRIGID benchmark run directories into comparable baseline tables.

The script is intentionally lightweight: it does not import FRIGID, RDKit,
torch, or any model code. It only reads completed run artifacts such as
aggregate_statistics.json, oracle_aggregate_statistics.json, run_manifest.json,
and optional QA summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


STANDARD_METRICS = [
    "total_spectra",
    "elapsed_time_seconds",
    "spectra_per_second",
    "exact_match_top1",
    "exact_match_top10",
    "tanimoto_top1_mean",
    "tanimoto_top10_mean",
    "mist_tanimoto_mean",
    "avg_formula_matches",
    "avg_predictions_collected",
    "avg_total_generated",
    "formula_match_success_rate",
    "avg_attempts_to_match",
    "never_matched_rate",
]

QA_METRICS = [
    "manifest_test_samples",
    "missing_test_objects",
    "missing_test_object_share",
    "exact_top1_missing_as_failures",
    "exact_top10_missing_as_failures",
    "proposal_equals_pred_smiles_1_share",
    "broken_symlinks",
]


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Run name cannot be empty")
    return name, Path(raw_path).expanduser()


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON file: {path}: {exc}") from exc


def find_first(run_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        direct = run_dir / name
        if direct.is_file():
            return direct
    for name in names:
        matches = sorted(run_dir.rglob(name))
        if matches:
            return matches[0]
    return None


def count_csv_rows(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def summarize_run(name: str, run_dir: Path, qa_summary: Path | None) -> dict[str, Any]:
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist for {name}: {run_dir}")

    aggregate_path = find_first(run_dir, ["aggregate_statistics.json", "oracle_aggregate_statistics.json"])
    if aggregate_path is None:
        raise SystemExit(f"No aggregate statistics JSON found for {name}: {run_dir}")

    manifest_path = find_first(run_dir, ["run_manifest.json"])
    detailed_path = find_first(run_dir, ["detailed_results.csv"])
    predictions_path = find_first(run_dir, ["predictions.csv", "oracle_predictions.csv"])

    aggregate = load_json(aggregate_path)
    manifest = load_json(manifest_path) if manifest_path else None
    qa = load_json(qa_summary) if qa_summary else None

    metrics = {key: aggregate.get(key) for key in STANDARD_METRICS if key in aggregate}
    paper_targets = {
        key: aggregate[key]
        for key in aggregate
        if key.startswith("paper_target") or key in {"paper_target_top1", "paper_target_top10"}
    }

    summary: dict[str, Any] = {
        "name": name,
        "run_dir": str(run_dir),
        "aggregate_statistics": str(aggregate_path),
        "run_manifest": str(manifest_path) if manifest_path else None,
        "detailed_results": str(detailed_path) if detailed_path else None,
        "predictions": str(predictions_path) if predictions_path else None,
        "detailed_result_rows": count_csv_rows(detailed_path),
        "prediction_rows": count_csv_rows(predictions_path),
        "metrics": metrics,
        "paper_targets": paper_targets,
    }

    if manifest:
        summary["manifest"] = {
            "mode": manifest.get("mode"),
            "scaler": manifest.get("scaler"),
            "status": manifest.get("status"),
            "repo": manifest.get("repo"),
            "inputs": manifest.get("inputs"),
            "parameters": manifest.get("parameters"),
        }

    if qa:
        summary["qa_summary"] = {key: qa.get(key) for key in QA_METRICS if key in qa}

    return summary


def write_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = sorted({key for item in summaries for key in item["metrics"]})
    qa_keys = sorted({key for item in summaries for key in item.get("qa_summary", {})})
    fieldnames = [
        "name",
        "run_dir",
        "aggregate_statistics",
        "detailed_result_rows",
        "prediction_rows",
        *metric_keys,
        *[f"qa_{key}" for key in qa_keys],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in summaries:
            row = {
                "name": item["name"],
                "run_dir": item["run_dir"],
                "aggregate_statistics": item["aggregate_statistics"],
                "detailed_result_rows": item["detailed_result_rows"],
                "prediction_rows": item["prediction_rows"],
            }
            row.update(item["metrics"])
            row.update({f"qa_{key}": value for key, value in item.get("qa_summary", {}).items()})
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize completed FRIGID benchmark runs into JSON and CSV baselines."
    )
    parser.add_argument("--run", action="append", type=parse_named_path, required=True, metavar="NAME=PATH")
    parser.add_argument("--qa-summary", action="append", type=parse_named_path, default=[], metavar="NAME=PATH")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    qa_by_name = dict(args.qa_summary)
    summaries = [summarize_run(name, run_dir, qa_by_name.get(name)) for name, run_dir in args.run]

    payload = {
        "schema_version": 1,
        "description": "Comparable FRIGID benchmark baseline summary generated from completed run artifacts.",
        "runs": summaries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(args.output_csv, summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
