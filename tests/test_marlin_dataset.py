import hashlib
import json
from pathlib import Path

import datasets
import pytest
import torch

import marlin.dataset as dataset_module
from marlin.dataset import (
    RankShardedIterableDataset,
    file_list_sha256,
    resolve_stream_offset_examples,
    sha256_file,
    streaming_loader_workers,
    verify_filtered_prefix_cache,
    verify_shuffled_stream_cache,
    verify_snapshot_manifest,
)


def test_rank_sharded_iterable_dataset_is_disjoint_and_complete() -> None:
    source = list(range(12))
    rank_zero = list(
        RankShardedIterableDataset(source, rank=0, world_size=2)
    )
    rank_one = list(
        RankShardedIterableDataset(source, rank=1, world_size=2)
    )

    assert rank_zero == list(range(0, 12, 2))
    assert rank_one == list(range(1, 12, 2))
    assert sorted(rank_zero + rank_one) == source
    assert set(rank_zero).isdisjoint(rank_one)


def test_stream_offset_is_derived_from_canonical_checkpoint_step() -> None:
    assert resolve_stream_offset_examples(
        resume_checkpoint="/runs/checkpoints/step=500.ckpt",
        batch_size=8,
        devices=2,
        accumulate_grad_batches=16,
    ) == 128_000


def test_stream_offset_requires_explicit_value_for_noncanonical_checkpoint() -> None:
    with pytest.raises(ValueError, match="non-canonical checkpoint"):
        resolve_stream_offset_examples(
            resume_checkpoint="/runs/checkpoints/last.ckpt",
            batch_size=8,
            devices=2,
            accumulate_grad_batches=16,
        )

    assert resolve_stream_offset_examples(
        resume_checkpoint="/runs/checkpoints/last.ckpt",
        batch_size=8,
        devices=2,
        accumulate_grad_batches=16,
        explicit_offset=12_345,
    ) == 12_345


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


def test_verify_shuffled_stream_cache_returns_shard_and_rows(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "shuffled-stream.parquet"
    shard.write_bytes(b"cached-shuffled-stream")
    manifest = {
        "schema_version": 1,
        "kind": "MARLIN shuffled eligible stream cache",
        "source_manifest_sha256": "source",
        "filtered_prefix_manifest_sha256": "prefix",
        "tokenizer_sha256": "tokenizer",
        "exclusion_sha256": "exclusions",
        "max_length": 256,
        "seed": 42,
        "shuffle_buffer": 100,
        "rows": 1000,
        "shard": shard.name,
        "shard_size_bytes": shard.stat().st_size,
        "shard_sha256": sha256_file(shard),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    assert verify_shuffled_stream_cache(
        path,
        source_manifest_sha256="source",
        filtered_prefix_manifest_sha256="prefix",
        tokenizer_sha256="tokenizer",
        exclusion_sha256="exclusions",
        max_length=256,
        seed=42,
        shuffle_buffer=100,
        minimum_rows=900,
    ) == (str(shard), 1000)


def test_verify_shuffled_stream_cache_rejects_short_cache(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "shuffled-stream.parquet"
    shard.write_bytes(b"short")
    manifest = {
        "schema_version": 1,
        "kind": "MARLIN shuffled eligible stream cache",
        "source_manifest_sha256": "source",
        "filtered_prefix_manifest_sha256": "prefix",
        "tokenizer_sha256": "tokenizer",
        "exclusion_sha256": "exclusions",
        "max_length": 256,
        "seed": 42,
        "shuffle_buffer": 100,
        "rows": 10,
        "shard": shard.name,
        "shard_size_bytes": shard.stat().st_size,
        "shard_sha256": sha256_file(shard),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="100 are required"):
        verify_shuffled_stream_cache(
            path,
            source_manifest_sha256="source",
            filtered_prefix_manifest_sha256="prefix",
            tokenizer_sha256="tokenizer",
            exclusion_sha256="exclusions",
            max_length=256,
            seed=42,
            shuffle_buffer=100,
            minimum_rows=100,
        )


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


def test_verify_snapshot_manifest_reuses_identity_bound_hash_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard = tmp_path / "data/train/train-00000.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"verified")
    files = [
        {
            "path": str(shard.relative_to(tmp_path)),
            "size_bytes": shard.stat().st_size,
            "sha256": sha256_file(shard),
        }
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "datamol-io/safe-gpt",
                "revision": "pinned",
                "snapshot_root": str(tmp_path),
                "files": files,
                "schema_version": 1,
                "kind": "MARLIN offline SAFE-GPT training snapshot",
                "file_count": 1,
                "total_size_bytes": shard.stat().st_size,
                "file_list_sha256": file_list_sha256(files),
            }
        )
    )
    arguments = {
        "expected_dataset": "datamol-io/safe-gpt",
        "expected_revision": "pinned",
        "expected_file_list_sha256": file_list_sha256(files),
    }

    verify_snapshot_manifest(manifest, **arguments)
    original_sha256_file = dataset_module.sha256_file

    def reject_shard_rehash(path: Path) -> str:
        if path.suffix == ".parquet":
            raise AssertionError("verified shard was hashed again")
        return original_sha256_file(path)

    monkeypatch.setattr(dataset_module, "sha256_file", reject_shard_rehash)
    verify_snapshot_manifest(manifest, **arguments)


def test_verify_snapshot_manifest_invalidates_cache_after_shard_change(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "data/train/train-00000.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"original")
    files = [
        {
            "path": str(shard.relative_to(tmp_path)),
            "size_bytes": shard.stat().st_size,
            "sha256": sha256_file(shard),
        }
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "datamol-io/safe-gpt",
                "revision": "pinned",
                "snapshot_root": str(tmp_path),
                "files": files,
                "schema_version": 1,
                "kind": "MARLIN offline SAFE-GPT training snapshot",
                "file_count": 1,
                "total_size_bytes": shard.stat().st_size,
                "file_list_sha256": file_list_sha256(files),
            }
        )
    )
    arguments = {
        "expected_dataset": "datamol-io/safe-gpt",
        "expected_revision": "pinned",
        "expected_file_list_sha256": file_list_sha256(files),
    }

    verify_snapshot_manifest(manifest, **arguments)
    shard.write_bytes(b"modified")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_snapshot_manifest(manifest, **arguments)


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
