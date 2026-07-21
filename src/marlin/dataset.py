"""Offline, content-addressed MARLIN training dataset helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot_manifest(
    manifest_path: str | Path,
    *,
    expected_dataset: str,
    expected_revision: str,
) -> tuple[list[str], str]:
    """Verify every local shard against the pinned repository manifest."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if manifest.get("dataset") != expected_dataset:
        raise ValueError("local training snapshot dataset mismatch")
    if manifest.get("revision") != expected_revision:
        raise ValueError("local training snapshot revision mismatch")
    root = Path(manifest["snapshot_root"])
    files = manifest.get("files", [])
    if not files:
        raise ValueError("local training snapshot manifest has no shards")
    resolved = []
    for entry in files:
        shard = root / entry["path"]
        if not shard.is_file():
            raise FileNotFoundError(f"missing local training shard {shard}")
        if shard.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"local training shard size mismatch: {shard}")
        observed = sha256_file(shard)
        if observed != entry["sha256"]:
            raise ValueError(f"local training shard SHA-256 mismatch: {shard}")
        resolved.append(str(shard.resolve()))
    if resolved != sorted(resolved):
        raise ValueError("local training shards are not in deterministic order")
    return resolved, sha256_file(path)
