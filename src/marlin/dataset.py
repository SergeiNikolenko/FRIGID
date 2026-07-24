"""Offline, content-addressed MARLIN training dataset helpers."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_list_sha256(files: list[dict[str, object]]) -> str:
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_snapshot_manifest(
    manifest_path: str | Path,
    *,
    expected_dataset: str,
    expected_revision: str,
    expected_file_list_sha256: str,
    verify_hashes: bool = True,
) -> tuple[list[str], str]:
    """Verify local shards against the pinned repository manifest."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("local training snapshot schema mismatch")
    if manifest.get("kind") != "MARLIN offline SAFE-GPT training snapshot":
        raise ValueError("local training snapshot kind mismatch")
    if manifest.get("dataset") != expected_dataset:
        raise ValueError("local training snapshot dataset mismatch")
    if manifest.get("revision") != expected_revision:
        raise ValueError("local training snapshot revision mismatch")
    root = Path(manifest["snapshot_root"]).resolve()
    files = manifest.get("files", [])
    if not files:
        raise ValueError("local training snapshot manifest has no shards")
    if manifest.get("file_count") != len(files):
        raise ValueError("local training snapshot file count mismatch")
    if manifest.get("total_size_bytes") != sum(entry["size_bytes"] for entry in files):
        raise ValueError("local training snapshot total size mismatch")
    if files != sorted(files, key=lambda entry: entry["path"]):
        raise ValueError("local training snapshot entries are not sorted")
    paths = [entry["path"] for entry in files]
    if len(paths) != len(set(paths)):
        raise ValueError("local training snapshot contains duplicate shards")
    observed_list_sha256 = file_list_sha256(files)
    if manifest.get("file_list_sha256") != observed_list_sha256:
        raise ValueError("local training snapshot embedded file-list digest mismatch")
    if observed_list_sha256 != expected_file_list_sha256:
        raise ValueError("local training snapshot canonical file-list digest mismatch")
    resolved = []
    for entry in files:
        relative = PurePosixPath(entry["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not re.fullmatch(r"data/train/train-\d{5}\.parquet", entry["path"])
        ):
            raise ValueError(f"invalid local training shard path: {entry['path']}")
        shard = (root / Path(*relative.parts)).resolve()
        if not shard.is_relative_to(root):
            raise ValueError(f"local training shard escapes snapshot root: {shard}")
        if not shard.is_file():
            raise FileNotFoundError(f"missing local training shard {shard}")
        if shard.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"local training shard size mismatch: {shard}")
        if verify_hashes:
            observed = sha256_file(shard)
            if observed != entry["sha256"]:
                raise ValueError(f"local training shard SHA-256 mismatch: {shard}")
        resolved.append(str(shard.resolve()))
    return resolved, sha256_file(path)
