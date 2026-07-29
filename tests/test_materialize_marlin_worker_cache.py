import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1] / "scripts/materialize_marlin_worker_cache.py"
)
SPEC = importlib.util.spec_from_file_location(
    "materialize_marlin_worker_cache", SCRIPT
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_extract_frigid_checkpoint_verifies_content(tmp_path, monkeypatch):
    payload = b"pinned FRIGID checkpoint"
    archive = tmp_path / "checkpoints.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo("DLM.ckpt")
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    monkeypatch.setattr(
        MODULE,
        "FRIGID_CHECKPOINT_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )

    checkpoint = tmp_path / "frigid" / "DLM.ckpt"
    MODULE.extract_frigid_checkpoint(archive, checkpoint)

    assert checkpoint.read_bytes() == payload
    assert checkpoint.stat().st_mode & 0o777 == 0o640


def test_archive_download_resumes_partial_file(tmp_path, monkeypatch):
    payload = b"complete archive payload"
    destination = tmp_path / "archive.tar.gz"
    partial = destination.with_suffix(".gz.part")
    partial.write_bytes(payload[:8])

    class Response(io.BytesIO):
        status = 206

        def getcode(self):
            return self.status

    monkeypatch.setattr(
        MODULE.urllib.request,
        "urlopen",
        lambda request, timeout: Response(payload[8:]),
    )

    MODULE.download_with_resume(
        "https://example.invalid/archive",
        destination,
        expected_size=len(payload),
        expected_md5=hashlib.md5(payload).hexdigest(),
        max_attempts=1,
    )

    assert destination.read_bytes() == payload
    assert not partial.exists()


def test_worker_cache_plan_does_not_materialize(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT),
            "--root",
            str(tmp_path / "cache"),
            "--plan",
        ],
    )

    MODULE.main()

    assert not (tmp_path / "cache").exists()
