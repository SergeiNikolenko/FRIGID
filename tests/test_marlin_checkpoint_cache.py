import hashlib
from pathlib import Path

import pytest

from scripts.materialize_marlin_checkpoint import (
    materialize_checkpoint,
)


def test_materialize_checkpoint_reuses_verified_cache(
    tmp_path: Path,
) -> None:
    payload = b"checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    checkpoint = tmp_path / digest / "checkpoint.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(payload)

    assert materialize_checkpoint(
        task_id="unused",
        artifact_name="unused",
        expected_sha256=digest,
        cache_root=tmp_path,
    ) == checkpoint


def test_materialize_checkpoint_rejects_corrupt_cache(
    tmp_path: Path,
) -> None:
    digest = "0" * 64
    checkpoint = tmp_path / digest / "checkpoint.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="cached checkpoint SHA-256"):
        materialize_checkpoint(
            task_id="unused",
            artifact_name="unused",
            expected_sha256=digest,
            cache_root=tmp_path,
        )


def test_materialize_checkpoint_preserves_explicit_resume_step(
    tmp_path: Path,
) -> None:
    payload = b"checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    checkpoint = tmp_path / digest / "step=2000.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(payload)

    assert materialize_checkpoint(
        task_id="unused",
        artifact_name="unused",
        expected_sha256=digest,
        cache_root=tmp_path,
        output_name="step=2000.ckpt",
    ) == checkpoint


def test_materialize_checkpoint_rejects_unsafe_output_name(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="output name"):
        materialize_checkpoint(
            task_id="unused",
            artifact_name="unused",
            expected_sha256="0" * 64,
            cache_root=tmp_path,
            output_name="../step=2000.ckpt",
        )


def test_materialize_checkpoint_downloads_from_verified_artifact_uri(
    monkeypatch,
    tmp_path: Path,
) -> None:
    payload = b"checkpoint from storage"
    digest = hashlib.sha256(payload).hexdigest()
    downloaded = tmp_path / "downloaded.ckpt"
    downloaded.write_bytes(payload)
    monkeypatch.setattr(
        "clearml.StorageManager.get_local_copy",
        lambda **kwargs: str(downloaded),
    )

    checkpoint = materialize_checkpoint(
        task_id=None,
        artifact_name=None,
        artifact_uri="https://files.example/checkpoint.ckpt",
        expected_sha256=digest,
        cache_root=tmp_path / "cache",
    )

    assert checkpoint.read_bytes() == payload
    assert checkpoint == tmp_path / "cache" / digest / "checkpoint.ckpt"


def test_materialize_checkpoint_hashes_and_caches_clearml_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    payload = b"checkpoint from output model"
    digest = hashlib.sha256(payload).hexdigest()
    downloaded = tmp_path / "downloaded.ckpt"
    downloaded.write_bytes(payload)

    class FakeModel:
        def __init__(self, *, model_id):
            assert model_id == "model-id"

        def get_local_copy(self):
            return str(downloaded)

    monkeypatch.setattr("clearml.Model", FakeModel)
    checkpoint = materialize_checkpoint(
        task_id=None,
        artifact_name=None,
        model_id="model-id",
        expected_sha256=None,
        cache_root=tmp_path / "cache",
        output_name="step=30000.ckpt",
    )

    assert checkpoint.read_bytes() == payload
    assert checkpoint == tmp_path / "cache" / digest / "step=30000.ckpt"
