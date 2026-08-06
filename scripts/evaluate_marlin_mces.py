#!/usr/bin/env python3
"""Compute paper-reported myopic MCES metrics from saved MARLIN predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threshold", type=int, default=15)
    parser.add_argument("--solver", default="PULP_CBC_CMD")
    parser.add_argument("--time-limit", type=int, default=600)
    parser.add_argument("--clearml-task-id")
    parser.add_argument("--clearml-iteration", type=int)
    return parser.parse_args()


def publish_clearml_mces(
    metrics: dict,
    *,
    task_id: str | None,
    iteration: int | None,
) -> None:
    """Attach paper MCES summaries to the same task as molecular generation."""
    if task_id is None:
        return
    from clearml import Task

    logger = Task.get_task(task_id=task_id).get_logger()
    report_iteration = int(metrics["rows"] if iteration is None else iteration)
    for series, key in (
        ("MCES@1 (returned)", "mces_top1"),
        ("MCES@10 (returned)", "mces_top10"),
    ):
        value = float(metrics[key])
        if np.isfinite(value):
            logger.report_scalar(
                title="Paper parity",
                series=series,
                value=value,
                iteration=report_iteration,
            )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_row(payload: tuple[dict, int, str, int]) -> dict:
    row, threshold, solver_name, time_limit = payload
    from myopic_mces import MCES

    candidates = row["candidates"][:10]
    distances = [
        float(
            MCES(
                row["target_smiles"],
                candidate["smiles"],
                solver=solver_name,
                solver_options={"msg": 0, "timeLimit": time_limit},
                threshold=threshold,
                always_stronger_bound=True,
            )[1]
        )
        for candidate in candidates
    ]
    return {
        "spec_name": row["spec_name"],
        "lane": row["lane"],
        "candidate_count": len(candidates),
        "mces_top1": distances[0] if distances else None,
        "mces_top10": min(distances) if distances else None,
    }


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    with args.predictions.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["spec_name"] in seen:
                raise ValueError(f"duplicate spectrum: {row['spec_name']}")
            seen.add(row["spec_name"])
            rows.append(row)

    signature = {
        "schema_version": 1,
        "predictions_sha256": sha256(args.predictions),
        "threshold": args.threshold,
        "solver": args.solver,
        "time_limit": args.time_limit,
        "myopic_mces_version": "1.2.0",
        "pulp_version": "3.3.2",
    }
    signature_path = args.output_dir / "mces_signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError(f"incompatible MCES run in {args.output_dir}")
    signature_path.write_text(json.dumps(signature, indent=2, sort_keys=True) + "\n")

    results_path = args.output_dir / "mces.jsonl"
    completed = {}
    if results_path.exists():
        with results_path.open() as handle:
            for line in handle:
                result = json.loads(line)
                if result["spec_name"] in completed:
                    raise ValueError(f"duplicate MCES result: {result['spec_name']}")
                completed[result["spec_name"]] = result
    pending = [row for row in rows if row["spec_name"] not in completed]
    payloads = [
        (row, args.threshold, args.solver, args.time_limit) for row in pending
    ]
    with results_path.open("a") as output, ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:
        for result in executor.map(compute_row, payloads, chunksize=1):
            output.write(json.dumps(result, sort_keys=True) + "\n")
            output.flush()
            completed[result["spec_name"]] = result
            print(
                f"{len(completed)}/{len(rows)} {result['spec_name']} "
                f"mces_top1={result['mces_top1']} mces_top10={result['mces_top10']}",
                flush=True,
            )

    returned = [
        completed[row["spec_name"]]
        for row in rows
        if completed[row["spec_name"]]["candidate_count"] > 0
    ]
    metrics = {
        "rows": len(rows),
        "rows_with_candidate": len(returned),
        "mces_top1": float(np.mean([row["mces_top1"] for row in returned]))
        if returned
        else float("nan"),
        "mces_top10": float(np.mean([row["mces_top10"] for row in returned]))
        if returned
        else float("nan"),
        "settings": signature,
    }
    (args.output_dir / "mces_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    publish_clearml_mces(
        metrics,
        task_id=args.clearml_task_id,
        iteration=args.clearml_iteration,
    )


if __name__ == "__main__":
    main()
