from marlin.dataset import file_list_sha256 as dataset_file_list_sha256
from marlin.dataset import sha256_file as dataset_sha256_file
from scripts.materialize_marlin_training_snapshot import (
    file_list_sha256,
    sha256_file,
)


def test_snapshot_hash_helpers_match_dataset_helpers(tmp_path):
    shard = tmp_path / "shard.parquet"
    shard.write_bytes(b"pinned snapshot")
    files = [
        {
            "path": "data/train/shard.parquet",
            "size_bytes": shard.stat().st_size,
            "sha256": dataset_sha256_file(shard),
        }
    ]

    assert sha256_file(shard) == dataset_sha256_file(shard)
    assert file_list_sha256(files) == dataset_file_list_sha256(files)
