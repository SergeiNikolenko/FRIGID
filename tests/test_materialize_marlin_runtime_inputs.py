import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_marlin_runtime_inputs.py"
)
SPEC = importlib.util.spec_from_file_location(
    "materialize_marlin_runtime_inputs",
    SCRIPT,
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _bundle(tmp_path: Path, *, extra_member: str | None = None) -> tuple[Path, str]:
    payloads = {
        "length_audit.json": b"length-audit",
        "nplib1_test_inchikeys.csv": b"inchikey\nAAAA\n",
        "tokenizer.json": b"tokenizer",
        "val/fingerprints.npz": b"fingerprints",
        "val/metadata.csv": b"smiles\nCC\n",
    }
    manifest = {
        "schema_version": 1,
        "kind": "MARLIN FARO runtime input bundle",
        "files": [
            {
                "path": path,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            for path, payload in sorted(payloads.items())
        ],
    }
    bundle = tmp_path / "inputs.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for name, payload in {
            **payloads,
            "manifest.json": json.dumps(manifest).encode(),
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        if extra_member:
            payload = b"unexpected"
            info = tarfile.TarInfo(extra_member)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return bundle, MODULE.sha256_file(bundle)


def test_materialize_bundle_is_atomic_and_revalidates_existing_inputs(tmp_path):
    bundle, digest = _bundle(tmp_path)
    output = tmp_path / "runtime"

    manifest = MODULE.materialize_bundle(
        bundle,
        output,
        expected_sha256=digest,
    )
    assert manifest["kind"] == "MARLIN FARO runtime input bundle"
    assert (output / "val/metadata.csv").read_text() == "smiles\nCC\n"

    MODULE.materialize_bundle(bundle, output, expected_sha256=digest)
    (output / "tokenizer.json").write_text("changed")
    with pytest.raises(ValueError, match="size mismatch"):
        MODULE.materialize_bundle(bundle, output, expected_sha256=digest)


@pytest.mark.parametrize("member", ("../escape", "/absolute", "extra.txt"))
def test_materialize_bundle_rejects_unsafe_or_unexpected_members(
    tmp_path,
    member,
):
    bundle, digest = _bundle(tmp_path, extra_member=member)

    with pytest.raises(ValueError, match="unsafe|unexpected"):
        MODULE.materialize_bundle(
            bundle,
            tmp_path / "runtime",
            expected_sha256=digest,
        )
