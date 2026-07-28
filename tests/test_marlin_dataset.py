import hashlib
import json
from pathlib import Path

import datasets
import pytest
import torch

from marlin.dataset import (
    file_list_sha256,
    sha256_file,
    streaming_loader_workers,
    verify_filtered_prefix_cache,
    verify_snapshot_manifest,
)


def test_filtered_prefix_stream_disables_workers_for_unshardable_tail() -> None:
    assert streaming_loader_workers(8, uses_filtered_prefix=True) == 0
    assert streaming_loader_workers(0, uses_filtered_prefix=True) == 0
    assert streaming_loader_workers(8, uses_filtered_prefix=False) == 8


def test_filtered_prefix_and_skipped_tail_iterate_with_selected_workers() -> None:
    prefix = datasets.Dataset.from_dict({"safe": ["cached"]}).to_iterable_dataset()
    source = datasets.Dataset.from_dict(
        {"safe": ["raw-prefix", "tail"]}
    ).to_iterable_dataset()
    stream = datasets.concatenate_datasets([prefix, source.skip(1)])
    loader = torch.utils.data.DataLoader(
        stream,
        batch_size=1,
        num_workers=streaming_loader_workers(8, uses_filtered_prefix=True),
    )

    assert [batch["safe"][0] for batch in loader] == ["cached", "tail"]


def test_verify_filtered_prefix_cache_returns_shard_and_offset(tmp_path: Path) -> None:
    shard = tmp_path / "eligible-prefix.parquet"
    shard.write_bytes(b"cached-safe-prefix")
    manifest = {
        "schema_version": 1,
        "kind": "MARLIN filtered SAFE prefix cache",
        "source_manifest_sha256": "source",
        "tokenizer_sha256": "tokenizer",
        "exclusion_sha256": "exclusions",
        "max_length": 256,
        "eligible_rows": 100,
        "raw_rows_consumed": 123,
        "shard": shard.name,
        "shard_size_bytes": shard.stat().st_size,
        "shard_sha256": sha256_file(shard),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    assert verify_filtered_prefix_cache(
        path,
        source_manifest_sha256="source",
        tokenizer_sha256="tokenizer",
        exclusion_sha256="exclusions",
        max_length=256,
        minimum_rows=100,
    ) == (str(shard), 123)


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
                "schema_version": 1,
                "kind": "MARLIN offline SAFE-GPT training snapshot",
                "file_count": 2,
                "total_size_bytes": 11,
                "file_list_sha256": file_list_sha256(shards),
            }
        )
    )

    files, manifest_sha = verify_snapshot_manifest(
        manifest,
        expected_dataset="datamol-io/safe-gpt",
        expected_revision="pinned",
        expected_file_list_sha256=file_list_sha256(shards),
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
                "schema_version": 1,
                "kind": "MARLIN offline SAFE-GPT training snapshot",
                "files": [
                    {
                        "path": "data/train/train-00000.parquet",
                        "size_bytes": 7,
                        "sha256": "0" * 64,
                    }
                ],
                "file_count": 1,
                "total_size_bytes": 7,
                "file_list_sha256": file_list_sha256(
                    [
                        {
                            "path": "data/train/train-00000.parquet",
                            "size_bytes": 7,
                            "sha256": "0" * 64,
                        }
                    ]
                ),
            }
        )
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_snapshot_manifest(
            manifest,
            expected_dataset="datamol-io/safe-gpt",
            expected_revision="pinned",
            expected_file_list_sha256=file_list_sha256(
                [
                    {
                        "path": "data/train/train-00000.parquet",
                        "size_bytes": 7,
                        "sha256": "0" * 64,
                    }
                ]
            ),
        )


@pytest.mark.parametrize("bad_path", ["/tmp/external.parquet", "../escape.parquet"])
def test_verify_snapshot_manifest_rejects_non_relative_shard(
    tmp_path: Path, bad_path: str
) -> None:
    files = [{"path": bad_path, "size_bytes": 1, "sha256": "0" * 64}]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "MARLIN offline SAFE-GPT training snapshot",
                "dataset": "datamol-io/safe-gpt",
                "revision": "pinned",
                "snapshot_root": str(tmp_path),
                "files": files,
                "file_count": 1,
                "total_size_bytes": 1,
                "file_list_sha256": file_list_sha256(files),
            }
        )
    )

    with pytest.raises(ValueError, match="invalid local training shard path"):
        verify_snapshot_manifest(
            manifest,
            expected_dataset="datamol-io/safe-gpt",
            expected_revision="pinned",
            expected_file_list_sha256=file_list_sha256(files),
        )


def test_verify_snapshot_manifest_rejects_duplicate_shard(tmp_path: Path) -> None:
    entry = {
        "path": "data/train/train-00000.parquet",
        "size_bytes": 1,
        "sha256": "0" * 64,
    }
    files = [entry, dict(entry)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "MARLIN offline SAFE-GPT training snapshot",
                "dataset": "datamol-io/safe-gpt",
                "revision": "pinned",
                "snapshot_root": str(tmp_path),
                "files": files,
                "file_count": 2,
                "total_size_bytes": 2,
                "file_list_sha256": file_list_sha256(files),
            }
        )
    )

    with pytest.raises(ValueError, match="duplicate shards"):
        verify_snapshot_manifest(
            manifest,
            expected_dataset="datamol-io/safe-gpt",
            expected_revision="pinned",
            expected_file_list_sha256=file_list_sha256(files),
        )
