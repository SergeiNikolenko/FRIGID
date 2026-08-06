#!/usr/bin/env python3
"""Materialize and verify an immutable MARLIN checkpoint artifact."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id")
    parser.add_argument("--artifact-name")
    parser.add_argument("--artifact-uri")
    parser.add_argument("--model-id")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-name", default="checkpoint.ckpt")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_checkpoint(
    *,
    task_id: str | None,
    artifact_name: str | None,
    artifact_uri: str | None = None,
    model_id: str | None = None,
    expected_sha256: str | None,
    cache_root: Path,
    output_name: str = "checkpoint.ckpt",
) -> Path:
    digest = None
    if expected_sha256 is not None:
        if len(expected_sha256) != 64:
            raise ValueError("expected SHA-256 must contain 64 hexadecimal characters")
        digest = expected_sha256.lower()
        int(digest, 16)
    if not re.fullmatch(r"(?:checkpoint|step=\d+)\.ckpt", output_name):
        raise ValueError(
            "output name must be checkpoint.ckpt or step=<integer>.ckpt"
        )
    if digest is not None:
        destination = cache_root.resolve() / digest / output_name
        if destination.is_file():
            observed = sha256_file(destination)
            if observed != digest:
                raise ValueError(
                    f"cached checkpoint SHA-256 {observed} != {digest}"
                )
            return destination

    task_artifact_source = bool(task_id or artifact_name)
    if sum((bool(artifact_uri), bool(model_id), task_artifact_source)) != 1:
        raise ValueError(
            "set exactly one checkpoint source: task artifact, artifact URI, or model ID"
        )
    if artifact_uri:
        from clearml import StorageManager

        local_copy = StorageManager.get_local_copy(
            remote_url=artifact_uri,
            extract_archive=False,
        )
        if not local_copy:
            raise RuntimeError("ClearML storage manager did not return a local copy")
        downloaded = Path(local_copy)
    elif model_id:
        from clearml import Model

        local_copy = Model(model_id=model_id).get_local_copy()
        if not local_copy:
            raise RuntimeError("ClearML model did not return a local copy")
        downloaded = Path(local_copy)
    else:
        if not task_id or not artifact_name:
            raise ValueError(
                "task ID and artifact name are both required for task downloads"
            )
        from clearml import Task

        task = Task.get_task(task_id=task_id)
        if artifact_name not in task.artifacts:
            raise KeyError(f"task {task_id} has no artifact {artifact_name!r}")
        downloaded = Path(task.artifacts[artifact_name].get_local_copy())
    observed = sha256_file(downloaded)
    if digest is not None and observed != digest:
        raise ValueError(
            f"downloaded checkpoint SHA-256 {observed} != {digest}"
        )
    digest = digest or observed
    destination = cache_root.resolve() / digest / output_name
    if destination.is_file():
        cached = sha256_file(destination)
        if cached != digest:
            raise ValueError(f"cached checkpoint SHA-256 {cached} != {digest}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".checkpoint.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(downloaded, temporary)
        temporary.chmod(0o640)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main() -> None:
    args = parse_args()
    # The shell runner captures stdout as the resolved checkpoint path.
    # Keep SDK logging and transfer progress on stderr.
    with contextlib.redirect_stdout(sys.stderr):
        checkpoint = materialize_checkpoint(
            task_id=args.task_id,
            artifact_name=args.artifact_name,
            artifact_uri=args.artifact_uri,
            model_id=args.model_id,
            expected_sha256=args.expected_sha256,
            cache_root=args.cache_root,
            output_name=args.output_name,
        )
    print(checkpoint, flush=True)


if __name__ == "__main__":
    main()
