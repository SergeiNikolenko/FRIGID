import hashlib
import json
from pathlib import Path

import pytest

from marlin.dataset import verify_snapshot_manifest


def test_verify_snapshot_manifest_returns_ordered_verified_shards(tmp_path: Path) -> None:
    shards = []
    for name, content in (("train-00000.parquet", b"first"), ("train-00001.parquet", b"second")):
        path = tmp_path / "data/train" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        shards.append(
            {
                "path": str(path.relative_to(tmp_path)),
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "datamol-io/safe-gpt",
                "revision": "pinned",
                "snapshot_root": str(tmp_path),
                "files": shards,
            }
        )
    )

    files, manifest_sha = verify_snapshot_manifest(
        manifest,
        expected_dataset="datamol-io/safe-gpt",
        expected_revision="pinned",
    )

    assert files == sorted(files)
    assert manifest_sha == hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_verify_snapshot_manifest_rejects_corrupt_shard(tmp_path: Path) -> None:
    shard = tmp_path / "data/train/train-00000.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"corrupt")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "datamol-io/safe-gpt",
                "revision": "pinned",
                "snapshot_root": str(tmp_path),
                "files": [
                    {
                        "path": "data/train/train-00000.parquet",
                        "size_bytes": 7,
                        "sha256": "0" * 64,
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_snapshot_manifest(
            manifest,
            expected_dataset="datamol-io/safe-gpt",
            expected_revision="pinned",
        )
