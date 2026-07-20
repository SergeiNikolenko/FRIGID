#!/usr/bin/env python3
"""Export immutable ClearML scalar and artifact evidence for a training run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clearml import Task


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_evidence(reported: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for title, series_by_name in sorted(reported.items()):
        evidence[title] = {}
        for series, values in sorted(series_by_name.items()):
            x = list(values.get("x", []))
            y = list(values.get("y", []))
            evidence[title][series] = {
                "count": len(y),
                "x": x,
                "y": y,
            }
    return evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        type=Path,
        action="append",
        default=[],
        help="Local run artifact to hash; repeat for multiple files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = Task.get_task(task_id=args.task_id)
    artifacts = []
    for path in args.artifact:
        resolved = path.expanduser().resolve()
        artifacts.append(
            {
                "path": str(resolved),
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": {
            "id": task.id,
            "name": task.name,
            "project": task.get_project_name(),
            "status": task.status,
            "url": task.get_output_log_web_page(),
        },
        "reported_scalars": scalar_evidence(task.get_reported_scalars()),
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
