#!/usr/bin/env python
"""Evaluate paper-compatible thresholded myopic MCES on prediction shards."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


MCES_COLUMNS = tuple(f"mces@{k}" for k in range(1, 11))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_prediction_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Prediction input must use NAME=PATH form: {value!r}"
        )
    name, raw_path = value.split("=", maxsplit=1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Prediction input name cannot be empty")
    path = Path(raw_path.strip()).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Prediction CSV does not exist: {path}")
    return name, path


def load_expected_names(path: Path) -> list[str]:
    frame = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if "spec_name" not in frame:
        raise ValueError(f"Expected manifest is missing spec_name: {path}")
    names = frame["spec_name"].astype(str).tolist()
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("Expected manifest spec_name values must be non-empty and unique")
    return names


def load_prediction_frames(
    prediction_specs: list[tuple[str, Path]],
    expected_names: list[str],
) -> tuple[list[str], dict[str, pd.DataFrame]]:
    names = [name for name, _ in prediction_specs]
    if len(names) != len(set(names)):
        raise ValueError(f"Prediction variant names must be unique: {names}")
    required = {"name", "true_smiles", *(f"pred_smiles_{k}" for k in range(1, 11))}
    frames: dict[str, pd.DataFrame] = {}
    reference_targets: list[str] | None = None
    for name, path in prediction_specs:
        frame = pd.read_csv(path, dtype=str).fillna("")
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Prediction CSV {path} is missing columns: {missing}")
        observed = frame["name"].astype(str).tolist()
        if observed != expected_names:
            raise ValueError(
                f"Prediction CSV does not match locked manifest order: {path}"
            )
        targets = frame["true_smiles"].astype(str).tolist()
        if any(not target for target in targets):
            raise ValueError(f"Prediction CSV contains empty true_smiles: {path}")
        if reference_targets is None:
            reference_targets = targets
        elif targets != reference_targets:
            raise ValueError("Prediction variants contain different target SMILES")
        frames[name] = frame
    return names, frames


def load_runtime(
    manifest_path: Path,
    expected_sha256: str,
) -> tuple[dict[str, Any], Callable[..., tuple]]:
    actual_sha256 = sha256_file(manifest_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"MCES runtime manifest SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("purpose") != "frigid_mces_runtime":
        raise ValueError("Unexpected MCES runtime purpose")
    if manifest.get("status") != "completed":
        raise ValueError("MCES runtime is not completed")
    versions = manifest.get("overlay", {}).get("versions", {})
    if versions != {"PuLP": "2.7.0", "myopic-mces": "1.0.1"}:
        raise ValueError(f"Unexpected MCES runtime versions: {versions}")
    solver = manifest.get("solver", {})
    solver_path = Path(str(solver.get("path", ""))).expanduser().resolve()
    if solver.get("selected") != "PULP_CBC_CMD" or not solver_path.is_file():
        raise ValueError("Pinned PULP_CBC_CMD solver is unavailable")
    if sha256_file(solver_path) != solver.get("sha256"):
        raise ValueError("Pinned CBC binary SHA-256 mismatch")

    site_packages = Path(
        manifest.get("overlay", {}).get("site_packages", "")
    ).expanduser().resolve()
    if not site_packages.is_dir():
        raise ValueError(f"MCES overlay does not exist: {site_packages}")
    scripts_dir = Path(__file__).resolve().parent
    for path in (str(site_packages), str(scripts_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(site_packages), str(scripts_dir), existing_pythonpath]
    ).rstrip(os.pathsep)

    pulp = importlib.import_module("pulp")
    importlib.import_module("myopic_mces")
    if importlib.metadata.version("PuLP") != "2.7.0":
        raise ValueError("Loaded PuLP version differs from runtime manifest")
    if importlib.metadata.version("myopic-mces") != "1.0.1":
        raise ValueError("Loaded myopic-mces version differs from runtime manifest")
    if "PULP_CBC_CMD" not in pulp.listSolvers(onlyAvailable=True):
        raise ValueError("PULP_CBC_CMD is not available in the loaded runtime")
    metric_module = importlib.import_module("multi_compute")
    return manifest, metric_module.compute_metrics_for_one


def git_state(project_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "-C", project_root, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            [
                "git",
                "-C",
                project_root,
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def run_metric_task(
    metric_function: Callable[..., tuple],
    spec_name: str,
    variant: str,
    true_smiles: str,
    predictions: list[str],
) -> dict[str, Any]:
    metrics, _, _, _, _ = metric_function(
        true_smiles,
        predictions,
        solver="PULP_CBC_CMD",
        doMCES=True,
        doFull=False,
        filter_formula=False,
    )
    row: dict[str, Any] = {"spec_name": spec_name, "variant": variant}
    for column in MCES_COLUMNS:
        value = float(metrics[column])
        if not np.isfinite(value):
            raise ValueError(f"Non-finite {column} for {variant}/{spec_name}")
        row[column] = value
    return row


def write_partial(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def evaluate_shard(
    *,
    prediction_specs: list[tuple[str, Path]],
    expected_manifest: Path,
    runtime_manifest_path: Path,
    expected_runtime_manifest_sha256: str,
    output_dir: Path,
    start_index: int,
    max_spectra: int,
    n_jobs: int,
    query_batch_size: int,
    metric_function: Callable[..., tuple] | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError(f"MCES output directory already exists: {output_dir}")
    if start_index < 0 or max_spectra <= 0:
        raise ValueError("start_index must be non-negative and max_spectra positive")
    if n_jobs <= 0 or query_batch_size <= 0:
        raise ValueError("n_jobs and query_batch_size must be positive")

    expected_names = load_expected_names(expected_manifest)
    stop_index = start_index + max_spectra
    if stop_index > len(expected_names):
        raise ValueError("MCES shard range exceeds the locked manifest")
    variant_names, frames = load_prediction_frames(prediction_specs, expected_names)
    runtime_manifest: dict[str, Any]
    if metric_function is None:
        runtime_manifest, metric_function = load_runtime(
            runtime_manifest_path, expected_runtime_manifest_sha256
        )
    else:
        runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))

    project_root = Path(__file__).resolve().parents[1]
    commit, dirty = git_state(project_root)
    if dirty:
        raise ValueError(f"MCES evaluation checkout must be clean: {project_root}")

    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "per_sample_mces.csv"
    manifest_path = output_dir / "RUN_MANIFEST.json"
    selected_names = expected_names[start_index:stop_index]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "frigid_full_mces_shard",
        "status": "running",
        "start_timestamp": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
            "partition": os.environ.get("SLURM_JOB_PARTITION", "unknown"),
            "job_gpus": os.environ.get("SLURM_JOB_GPUS", "unknown"),
        },
        "code": {"commit": commit, "dirty": False},
        "selection": {
            "start_index": start_index,
            "max_spectra": max_spectra,
            "expected_total": len(expected_names),
        },
        "inputs": {
            "expected_manifest": {
                "path": str(expected_manifest),
                "sha256": sha256_file(expected_manifest),
            },
            "runtime_manifest": {
                "path": str(runtime_manifest_path),
                "sha256": expected_runtime_manifest_sha256,
            },
            "predictions": [
                {"name": name, "path": str(path), "sha256": sha256_file(path)}
                for name, path in prediction_specs
            ],
        },
        "runtime": {
            "versions": runtime_manifest.get("overlay", {}).get("versions", {}),
            "solver": runtime_manifest.get("solver", {}),
        },
        "settings": {
            "variant_order": variant_names,
            "top_k": 10,
            "threshold": 15,
            "always_stronger_bound": True,
            "solver_time_limit_seconds": 600,
            "filter_formula": False,
            "n_jobs": n_jobs,
            "query_batch_size": query_batch_size,
            "metric_implementation": "multi_compute.compute_metrics_for_one",
            "target_fields_used_for_metrics_only": ["true_smiles"],
        },
        "progress": {"processed_spectra": 0, "expected_spectra": max_spectra},
        "outputs": {},
    }
    write_manifest(manifest_path, manifest)

    rows: list[dict[str, Any]] = []
    try:
        for offset in range(0, max_spectra, query_batch_size):
            batch_names = selected_names[offset : offset + query_batch_size]
            tasks = []
            for local_index, spec_name in enumerate(batch_names, start=start_index + offset):
                for variant in variant_names:
                    frame_row = frames[variant].iloc[local_index]
                    predictions = [
                        str(frame_row[f"pred_smiles_{k}"]).strip() or None
                        for k in range(1, 11)
                    ]
                    tasks.append(
                        (
                            spec_name,
                            variant,
                            str(frame_row["true_smiles"]),
                            predictions,
                        )
                    )
            batch_rows = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(run_metric_task)(metric_function, *task) for task in tasks
            )
            rows.extend(batch_rows)
            write_partial(output_path, rows)
            manifest["progress"]["processed_spectra"] = min(
                offset + len(batch_names), max_spectra
            )
            manifest["progress"]["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_manifest(manifest_path, manifest)

        expected_pairs = [
            (spec_name, variant)
            for spec_name in selected_names
            for variant in variant_names
        ]
        observed_pairs = [(row["spec_name"], row["variant"]) for row in rows]
        if observed_pairs != expected_pairs:
            raise ValueError("MCES output does not match the declared ordered shard")
        manifest["status"] = "completed"
        manifest["exit_code"] = 0
        manifest["end_timestamp"] = datetime.now(timezone.utc).isoformat()
        manifest["outputs"] = {
            "per_sample_mces": {
                "path": str(output_path),
                "sha256": sha256_file(output_path),
                "row_count": len(rows),
            }
        }
        write_manifest(manifest_path, manifest)
        return manifest
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["exit_code"] = 1
        manifest["end_timestamp"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_manifest(manifest_path, manifest)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        action="append",
        required=True,
        type=parse_prediction_argument,
    )
    parser.add_argument("--expected-manifest", required=True, type=Path)
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--expected-runtime-manifest-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-index", required=True, type=int)
    parser.add_argument("--max-spectra", required=True, type=int)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--query-batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    evaluate_shard(
        prediction_specs=args.predictions,
        expected_manifest=args.expected_manifest.expanduser().resolve(),
        runtime_manifest_path=args.runtime_manifest.expanduser().resolve(),
        expected_runtime_manifest_sha256=args.expected_runtime_manifest_sha256,
        output_dir=args.output_dir.expanduser().resolve(),
        start_index=args.start_index,
        max_spectra=args.max_spectra,
        n_jobs=args.n_jobs,
        query_batch_size=args.query_batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
