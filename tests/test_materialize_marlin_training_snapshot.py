import builtins
import importlib.util
from pathlib import Path

from marlin.dataset import file_list_sha256 as dataset_file_list_sha256
from marlin.dataset import sha256_file as dataset_sha256_file


SCRIPT = (
    Path(__file__).parents[1] / "scripts/materialize_marlin_training_snapshot.py"
)


def load_snapshot_module():
    spec = importlib.util.spec_from_file_location(
        "materialize_marlin_training_snapshot", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_snapshot_hash_helpers_match_dataset_helpers(tmp_path):
    module = load_snapshot_module()
    shard = tmp_path / "shard.parquet"
    shard.write_bytes(b"pinned snapshot")
    files = [
        {
            "path": "data/train/shard.parquet",
            "size_bytes": shard.stat().st_size,
            "sha256": dataset_sha256_file(shard),
        }
    ]

    assert module.sha256_file(shard) == dataset_sha256_file(shard)
    assert module.file_list_sha256(files) == dataset_file_list_sha256(files)


def test_snapshot_materializer_import_does_not_require_torch(monkeypatch):
    original_import = builtins.__import__

    def reject_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("snapshot materialization must not import torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_torch)

    load_snapshot_module()
