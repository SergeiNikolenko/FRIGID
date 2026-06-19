"""Run artifact helpers for FRIGID command line tools."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def utc_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def file_md5(path: str | os.PathLike[str] | None) -> str | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.md5()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def git_dirty(repo_root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None
    return bool(result.stdout.strip())


def write_manifest(
    output_dir: str | os.PathLike[str],
    *,
    command: list[str],
    mode: str,
    scaler: str,
    repo_root: str | os.PathLike[str],
    inputs: dict[str, Any],
    checkpoints: dict[str, str | None],
    parameters: dict[str, Any],
    status: str,
    exit_code: int,
) -> Path:
    """Write a stable machine-readable run manifest."""
    root = Path(repo_root).resolve()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "created_at": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "mode": mode,
        "scaler": scaler,
        "status": status,
        "exit_code": exit_code,
        "command": command,
        "repo": {
            "root": str(root),
            "git_commit": git_commit(root),
            "git_dirty": git_dirty(root),
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "inputs": inputs,
        "checkpoints": {
            name: {
                "path": value,
                "md5": file_md5(value),
            }
            for name, value in checkpoints.items()
        },
        "parameters": parameters,
        "outputs": {
            "aggregate_statistics": str(out / "aggregate_statistics.json"),
            "detailed_results": str(out / "detailed_results.csv"),
            "predictions": str(out / "predictions.csv"),
            "config": str(out / "config.yaml"),
        },
    }

    path = out / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
