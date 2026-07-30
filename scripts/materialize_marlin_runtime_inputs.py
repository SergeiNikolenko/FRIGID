#!/usr/bin/env python3
"""Materialize the small, pinned MARLIN runtime inputs from ClearML."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

LEGACY_EXPECTED_FILES = {
    "length_audit.json",
    "nplib1_test_inchikeys.csv",
    "tokenizer.json",
    "val/fingerprints.npz",
    "val/metadata.csv",
}
END_TO_END_EXPECTED_FILES = LEGACY_EXPECTED_FILES | {
    "test/dreams_predictions.npz",
    "test/dreams_predictions.summary.json",
    "test/metadata.csv",
}
EXPECTED_FILE_SETS = (LEGACY_EXPECTED_FILES, END_TO_END_EXPECTED_FILES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_member_name(name: str) -> str:
    while name.startswith("./"):
        name = name[2:]
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe runtime bundle member: {name!r}")
    return path.as_posix()


def _validate_manifest(manifest: dict[str, object]) -> list[dict[str, object]]:
    if manifest.get("schema_version") != 1:
        raise ValueError("runtime bundle schema mismatch")
    if manifest.get("kind") != "MARLIN FARO runtime input bundle":
        raise ValueError("runtime bundle kind mismatch")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("runtime bundle file manifest is missing")
    paths = [str(entry.get("path")) for entry in files if isinstance(entry, dict)]
    file_set = set(paths)
    if len(paths) != len(files) or file_set not in EXPECTED_FILE_SETS:
        raise ValueError("runtime bundle file set mismatch")
    if len(paths) != len(set(paths)):
        raise ValueError("runtime bundle contains duplicate file entries")
    return files


def verify_materialized_inputs(output_dir: Path) -> dict[str, object]:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in _validate_manifest(manifest):
        relative = Path(str(entry["path"]))
        path = output_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"runtime input is missing: {path}")
        if path.stat().st_size != int(entry["size_bytes"]):
            raise ValueError(f"runtime input size mismatch: {path}")
        if sha256_file(path) != str(entry["sha256"]):
            raise ValueError(f"runtime input SHA-256 mismatch: {path}")
    return manifest


def materialize_bundle(
    bundle: Path,
    output_dir: Path,
    *,
    expected_sha256: str,
) -> dict[str, object]:
    observed_sha256 = sha256_file(bundle)
    if observed_sha256 != expected_sha256:
        raise ValueError(
            f"runtime bundle SHA-256 {observed_sha256} != {expected_sha256}"
        )
    if output_dir.exists():
        manifest = verify_materialized_inputs(output_dir)
        print(f"verified existing runtime inputs {output_dir}", flush=True)
        return manifest

    with tarfile.open(bundle, "r:gz") as archive:
        members = {}
        for member in archive.getmembers():
            if member.isdir() and member.name.rstrip("/") in {"", "."}:
                continue
            normalized = _normalized_member_name(member.name)
            if normalized in members:
                raise ValueError(
                    f"runtime bundle contains duplicate member: {normalized}"
                )
            members[normalized] = member
        manifest_member = members.get("manifest.json")
        if manifest_member is None or not manifest_member.isfile():
            raise ValueError("runtime bundle manifest is missing")
        manifest_source = archive.extractfile(manifest_member)
        if manifest_source is None:
            raise ValueError("runtime bundle manifest is unreadable")
        manifest = json.load(manifest_source)
        files = _validate_manifest(manifest)
        allowed_members = set(str(entry["path"]) for entry in files) | {
            "manifest.json"
        }
        unexpected = {
            name
            for name, member in members.items()
            if not member.isdir() and name not in allowed_members
        }
        if unexpected:
            raise ValueError(
                "runtime bundle contains unexpected members: "
                + ", ".join(sorted(unexpected))
            )

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=output_dir.parent,
            prefix=f".{output_dir.name}.",
        ) as temporary_text:
            temporary = Path(temporary_text)
            for entry in files:
                relative = str(entry["path"])
                member = members.get(relative)
                if member is None or not member.isfile():
                    raise ValueError(f"runtime bundle file is missing: {relative}")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"runtime bundle file is unreadable: {relative}")
                destination = temporary / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source, destination.open("wb") as target:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        target.write(chunk)
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            verify_materialized_inputs(temporary)
            temporary.rename(output_dir)
    print(f"runtime_inputs={output_dir} sha256={observed_sha256}", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-task-id",
        default=os.environ.get("MARLIN_RUNTIME_INPUT_TASK_ID"),
    )
    parser.add_argument(
        "--artifact-name",
        default=os.environ.get(
            "MARLIN_RUNTIME_INPUT_ARTIFACT",
            "runtime-inputs",
        ),
    )
    parser.add_argument(
        "--bundle-sha256",
        default=os.environ.get("MARLIN_RUNTIME_INPUT_BUNDLE_SHA256"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "MARLIN_RUNTIME_INPUT_ROOT",
                "/mnt/netstorage/nikolenko/marlin/runtime-inputs-v1",
            )
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source_task_id:
        raise ValueError("source task ID is required")
    if not args.bundle_sha256:
        raise ValueError("runtime bundle SHA-256 is required")
    from clearml import Task

    task = Task.get_task(task_id=args.source_task_id)
    artifact = task.artifacts.get(args.artifact_name)
    if artifact is None:
        raise KeyError(
            f"ClearML task {args.source_task_id} has no "
            f"{args.artifact_name!r} artifact"
        )
    bundle = Path(artifact.get_local_copy())
    materialize_bundle(
        bundle,
        args.output_dir.resolve(),
        expected_sha256=args.bundle_sha256,
    )


if __name__ == "__main__":
    main()
