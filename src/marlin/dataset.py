"""Offline, content-addressed MARLIN training dataset helpers."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import tempfile
from pathlib import Path, PurePosixPath

import torch


class RankShardedIterableDataset(torch.utils.data.IterableDataset):
    """Give every distributed rank a disjoint slice of one deterministic stream.

    Lightning intentionally does not inject a ``DistributedSampler`` for
    iterable datasets. Without this wrapper, every DDP rank iterates the same
    Hugging Face stream and the declared global batch size is overstated.
    """

    def __init__(
        self,
        dataset,
        *,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        super().__init__()
        if (rank is None) != (world_size is None):
            raise ValueError("rank and world_size must be provided together")
        if world_size is not None and (world_size <= 0 or not 0 <= rank < world_size):
            raise ValueError("invalid distributed rank/world size")
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def _distributed_position(self) -> tuple[int, int]:
        if self.rank is not None and self.world_size is not None:
            return self.rank, self.world_size
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def __iter__(self):
        rank, world_size = self._distributed_position()
        return itertools.islice(iter(self.dataset), rank, None, world_size)


def resolve_stream_offset_examples(
    *,
    resume_checkpoint: str | Path | None,
    batch_size: int,
    devices: int,
    accumulate_grad_batches: int,
    explicit_offset: int | None = None,
) -> int:
    """Resolve the deterministic global-stream offset for an exact resume."""

    if min(batch_size, devices, accumulate_grad_batches) <= 0:
        raise ValueError("global batch components must be positive")
    if explicit_offset is not None:
        if explicit_offset < 0:
            raise ValueError("stream offset must be non-negative")
        return explicit_offset
    if resume_checkpoint is None:
        return 0
    match = re.fullmatch(r"step=(\d+)\.ckpt", Path(resume_checkpoint).name)
    if match is None:
        raise ValueError(
            "cannot infer stream offset from a non-canonical checkpoint name; "
            "set data.stream_offset_examples explicitly"
        )
    global_step = int(match.group(1))
    return global_step * batch_size * devices * accumulate_grad_batches


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_list_sha256(files: list[dict[str, object]]) -> str:
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def streaming_loader_workers(requested: int, *, uses_filtered_prefix: bool) -> int:
    """Choose a worker count compatible with the composed streaming pipeline."""
    if requested < 0:
        raise ValueError("data loader worker count must be non-negative")
    if uses_filtered_prefix:
        # Hugging Face SkipExamplesIterable cannot shard the uncached tail.
        return 0
    return requested


def _snapshot_verification_cache_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(
        f".{manifest_path.name}.sha256-cache-v1.json"
    )


def _sample_sha256(path: Path, size_bytes: int) -> str:
    block_size = 64 * 1024
    offsets = sorted(
        {
            0,
            max(0, size_bytes // 2 - block_size // 2),
            max(0, size_bytes - block_size),
        }
    )
    digest = hashlib.sha256()
    digest.update(str(size_bytes).encode())
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            digest.update(str(offset).encode())
            digest.update(handle.read(block_size))
    return digest.hexdigest()


def _shard_identity(shard: Path, root: Path) -> dict[str, int | str]:
    stat = shard.stat()
    return {
        "path": shard.relative_to(root).as_posix(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "inode": stat.st_ino,
        "device": stat.st_dev,
        "sample_sha256": _sample_sha256(shard, stat.st_size),
    }


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


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
    manifest_bytes = path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
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
    identities = []
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
        identity = _shard_identity(shard, root)
        if identity["size_bytes"] != entry["size_bytes"]:
            raise ValueError(f"local training shard size mismatch: {shard}")
        identities.append(identity)
        resolved.append(str(shard.resolve()))

    cache_path = _snapshot_verification_cache_path(path)
    cache_hit = False
    if verify_hashes and cache_path.is_file():
        try:
            cache = json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            cache = {}
        cache_hit = cache == {
            "schema_version": 1,
            "kind": "MARLIN snapshot SHA-256 verification cache",
            "manifest_sha256": manifest_sha256,
            "file_list_sha256": expected_file_list_sha256,
            "files": identities,
        }

    if verify_hashes and not cache_hit:
        for entry, shard_text in zip(files, resolved):
            shard = Path(shard_text)
            observed = sha256_file(shard)
            if observed != entry["sha256"]:
                raise ValueError(f"local training shard SHA-256 mismatch: {shard}")
        try:
            _write_json_atomic(
                cache_path,
                {
                    "schema_version": 1,
                    "kind": "MARLIN snapshot SHA-256 verification cache",
                    "manifest_sha256": manifest_sha256,
                    "file_list_sha256": expected_file_list_sha256,
                    "files": identities,
                },
            )
        except OSError:
            # A read-only snapshot remains usable after full verification.
            pass
    return resolved, manifest_sha256


def verify_filtered_prefix_cache(
    manifest_path: str | Path,
    *,
    source_manifest_sha256: str,
    tokenizer_sha256: str,
    exclusion_sha256: str,
    max_length: int,
    minimum_rows: int,
) -> tuple[str, int]:
    """Verify a cached eligible prefix and return its shard and raw offset."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    expected = {
        "schema_version": 1,
        "kind": "MARLIN filtered SAFE prefix cache",
        "source_manifest_sha256": source_manifest_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "exclusion_sha256": exclusion_sha256,
        "max_length": max_length,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"filtered prefix cache {key} mismatch")
    eligible_rows = int(manifest.get("eligible_rows", 0))
    raw_rows_consumed = int(manifest.get("raw_rows_consumed", 0))
    if eligible_rows < minimum_rows or raw_rows_consumed < eligible_rows:
        raise ValueError("filtered prefix cache does not cover the shuffle buffer")
    shard = (path.parent / manifest["shard"]).resolve()
    if not shard.is_relative_to(path.parent.resolve()) or not shard.is_file():
        raise ValueError("filtered prefix cache shard is missing or escapes its root")
    if shard.stat().st_size != int(manifest["shard_size_bytes"]):
        raise ValueError("filtered prefix cache shard size mismatch")
    if sha256_file(shard) != manifest["shard_sha256"]:
        raise ValueError("filtered prefix cache shard SHA-256 mismatch")
    return str(shard), raw_rows_consumed
