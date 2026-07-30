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
