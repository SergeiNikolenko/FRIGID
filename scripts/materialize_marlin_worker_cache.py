#!/usr/bin/env python3
"""Materialize the pinned MARLIN inputs on a remote worker cache."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

FRIGID_ARCHIVE_URL = (
    "https://zenodo.org/api/records/19685145/files/"
    "frigid_pretrained_checkpoints.tar.gz/content"
)
FRIGID_ARCHIVE_SIZE = 2_804_076_946
FRIGID_ARCHIVE_MD5 = "1059c193b5f3dd7034079076e943389b"
FRIGID_CHECKPOINT_SHA256 = (
    "b6177c2d43448380aba80ff41c01461ea34ca2ca93b213986954c5afb7f0f457"
)
SAFE_GPT_DATASET = "datamol-io/safe-gpt"
SAFE_GPT_REVISION = "16d0be9ad6177ae683a32a86204530e8ee624a0f"
SAFE_GPT_FILE_LIST_SHA256 = (
    "ca5356076d4e6e4920a019d55af0a769b0984d736f3feb9799045f3c49244e6f"
)
SAFE_GPT_TOTAL_SIZE = 71_333_266_529
MINIMUM_FREE_BYTES = 100 * 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/netstorage/nikolenko/marlin"),
    )
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--plan", action="store_true")
    return parser.parse_args()


def digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_with_resume(
    url: str,
    destination: Path,
    *,
    expected_size: int,
    expected_md5: str,
    max_attempts: int,
) -> None:
    if (
        destination.is_file()
        and destination.stat().st_size == expected_size
        and digest_file(destination, "md5") == expected_md5
    ):
        print(f"verified existing archive {destination}", flush=True)
        return

    partial = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                status = getattr(response, "status", response.getcode())
                if offset and status != 206:
                    partial.replace(partial.with_suffix(".part.no-range"))
                    offset = 0
                mode = "ab" if offset and status == 206 else "wb"
                downloaded = offset
                last_report = time.monotonic()
                with partial.open(mode) as handle:
                    while chunk := response.read(8 * 1024 * 1024):
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if time.monotonic() - last_report >= 30:
                            print(
                                f"downloaded={downloaded}/{expected_size} "
                                f"path={partial}",
                                flush=True,
                            )
                            last_report = time.monotonic()
            if partial.stat().st_size != expected_size:
                raise ValueError(
                    f"archive size {partial.stat().st_size} != {expected_size}"
                )
            observed_md5 = digest_file(partial, "md5")
            if observed_md5 != expected_md5:
                raise ValueError(
                    f"archive MD5 {observed_md5} != {expected_md5}"
                )
            partial.replace(destination)
            print(f"archive={destination} md5={observed_md5}", flush=True)
            return
        except Exception as error:
            if attempt == max_attempts:
                raise
            delay = min(300, 10 * 2 ** (attempt - 1))
            print(
                f"archive retry in {delay}s attempt={attempt}/{max_attempts} "
                f"error={error}",
                flush=True,
            )
            time.sleep(delay)


def extract_frigid_checkpoint(archive: Path, checkpoint: Path) -> None:
    if (
        checkpoint.is_file()
        and digest_file(checkpoint, "sha256") == FRIGID_CHECKPOINT_SHA256
    ):
        print(f"verified existing checkpoint {checkpoint}", flush=True)
        return

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_suffix(".ckpt.tmp")
    with tarfile.open(archive, "r:gz") as bundle:
        member = bundle.getmember("DLM.ckpt")
        source = bundle.extractfile(member)
        if source is None:
            raise ValueError("DLM.ckpt is not a regular archive member")
        with source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    observed_sha256 = digest_file(temporary, "sha256")
    if observed_sha256 != FRIGID_CHECKPOINT_SHA256:
        raise ValueError(
            f"checkpoint SHA-256 {observed_sha256} != "
            f"{FRIGID_CHECKPOINT_SHA256}"
        )
    temporary.replace(checkpoint)
    checkpoint.chmod(0o640)
    print(f"checkpoint={checkpoint} sha256={observed_sha256}", flush=True)


def materialize_snapshot(
    repository_root: Path,
    snapshot: Path,
    *,
    max_attempts: int,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(repository_root / "scripts/materialize_marlin_training_snapshot.py"),
        "--dataset",
        SAFE_GPT_DATASET,
        "--revision",
        SAFE_GPT_REVISION,
        "--output-dir",
        str(snapshot),
        "--max-attempts",
        str(max_attempts),
    ]
    environment = os.environ.copy()
    source_root = str(repository_root / "src")
    environment["PYTHONPATH"] = (
        f"{source_root}:{environment['PYTHONPATH']}"
        if environment.get("PYTHONPATH")
        else source_root
    )
    subprocess.run(command, check=True, env=environment)
    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["revision"] != SAFE_GPT_REVISION:
        raise ValueError("SAFE-GPT manifest revision mismatch")
    if manifest["file_list_sha256"] != SAFE_GPT_FILE_LIST_SHA256:
        raise ValueError("SAFE-GPT manifest file-list digest mismatch")
    if manifest["total_size_bytes"] != SAFE_GPT_TOTAL_SIZE:
        raise ValueError("SAFE-GPT manifest total size mismatch")
    return manifest


def main() -> None:
    args = parse_args()
    if args.max_attempts < 1:
        raise ValueError("max attempts must be positive")
    root = args.root.resolve()
    archive = root / "checkpoints/frigid/frigid_pretrained_checkpoints.tar.gz"
    checkpoint = root / "checkpoints/frigid/DLM.ckpt"
    snapshot = root / f"safe-gpt-{SAFE_GPT_REVISION}"
    plan = {
        "root": str(root),
        "minimum_free_bytes": MINIMUM_FREE_BYTES,
        "frigid_archive": str(archive),
        "frigid_archive_size": FRIGID_ARCHIVE_SIZE,
        "frigid_archive_md5": FRIGID_ARCHIVE_MD5,
        "frigid_checkpoint": str(checkpoint),
        "frigid_checkpoint_sha256": FRIGID_CHECKPOINT_SHA256,
        "safe_gpt_snapshot": str(snapshot),
        "safe_gpt_revision": SAFE_GPT_REVISION,
        "safe_gpt_total_size": SAFE_GPT_TOTAL_SIZE,
        "safe_gpt_file_list_sha256": SAFE_GPT_FILE_LIST_SHA256,
    }
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.plan:
        return

    root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(root).free
    print(f"free_bytes={free_bytes}", flush=True)
    if free_bytes < MINIMUM_FREE_BYTES:
        raise OSError(
            f"worker cache has {free_bytes} free bytes; "
            f"{MINIMUM_FREE_BYTES} required"
        )

    lock_path = root / ".materialize.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        download_with_resume(
            FRIGID_ARCHIVE_URL,
            archive,
            expected_size=FRIGID_ARCHIVE_SIZE,
            expected_md5=FRIGID_ARCHIVE_MD5,
            max_attempts=args.max_attempts,
        )
        extract_frigid_checkpoint(archive, checkpoint)
        snapshot_manifest = materialize_snapshot(
            Path(__file__).resolve().parents[1],
            snapshot,
            max_attempts=args.max_attempts,
        )
        ready = {
            **plan,
            "snapshot_manifest": str(snapshot / "manifest.json"),
            "snapshot_manifest_sha256": digest_file(
                snapshot / "manifest.json", "sha256"
            ),
            "snapshot_file_count": snapshot_manifest["file_count"],
        }
        temporary = root / "worker_cache_ready.json.tmp"
        temporary.write_text(json.dumps(ready, indent=2, sort_keys=True) + "\n")
        temporary.replace(root / "worker_cache_ready.json")
        print(f"ready={root / 'worker_cache_ready.json'}", flush=True)


if __name__ == "__main__":
    main()
