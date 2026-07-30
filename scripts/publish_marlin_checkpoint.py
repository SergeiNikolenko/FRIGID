#!/usr/bin/env python3
"""Publish a worker-local MARLIN checkpoint as a verified ClearML artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from clearml import Task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--expected-sha256")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is unavailable: {checkpoint}")
    observed = sha256_file(checkpoint)
    if args.expected_sha256 and observed != args.expected_sha256.lower():
        raise ValueError(
            f"checkpoint SHA-256 {observed} != {args.expected_sha256.lower()}"
        )
    task_id = os.environ.get("CLEARML_TASK_ID")
    if not task_id:
        raise RuntimeError("CLEARML_TASK_ID is required")
    task = Task.get_task(task_id=task_id)
    uploaded = task.upload_artifact(
        name=args.artifact_name,
        artifact_object=checkpoint,
        metadata={
            "sha256": observed,
            "size_bytes": checkpoint.stat().st_size,
            "source_path": str(checkpoint),
        },
        wait_on_upload=True,
    )
    if not uploaded:
        raise RuntimeError("ClearML rejected the checkpoint artifact upload")
    print(
        json.dumps(
            {
                "artifact_name": args.artifact_name,
                "checkpoint": str(checkpoint),
                "sha256": observed,
                "size_bytes": checkpoint.stat().st_size,
                "task_id": task_id,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
