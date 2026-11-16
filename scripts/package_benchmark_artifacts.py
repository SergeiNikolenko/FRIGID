#!/usr/bin/env python3
"""Package a historical git archive without machine-specific filesystem links.

Keep the original archive beside the portable evidence package. Internal links
become relative; unavailable external dataset references remain in the manifest,
not as broken symlinks or fabricated data. This is an evidence archive, not a
self-contained training dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import posixpath
import tarfile
from pathlib import Path, PurePosixPath

ROOT = "docs/FRIGID_project"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(source: Path, destination: Path, source_commit: str) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    manifest = {
        "schema_version": 1,
        "kind": "portable_evidence_archive",
        "source_commit": source_commit,
        "original_archive_sha256": sha256(source),
        "dataset_complete": False,
        "files": [],
        "internal_links": [],
        "external_references": [],
    }
    with tarfile.open(source, "r:gz") as original:
        members = original.getmembers()
        known = {entry.name.rstrip("/") for entry in members}
        for entry in members:
            parts = PurePosixPath(entry.name).parts
            if entry.name.startswith("/") or ".." in parts or not (
                entry.name.rstrip("/") == "docs"
                or entry.name.rstrip("/") == ROOT
                or entry.name.startswith(ROOT + "/")
            ):
                raise ValueError(f"Unexpected archive path: {entry.name}")
            if not (entry.isfile() or entry.isdir() or entry.issym()):
                raise ValueError(f"Unsupported archive entry: {entry.name}")
        with tarfile.open(destination, "w:gz", compresslevel=6) as portable:
            for entry in members:
                if entry.issym():
                    marker = "/" + ROOT + "/"
                    target = (
                        ROOT + "/" + entry.linkname.split(marker, 1)[1]
                        if marker in entry.linkname
                        else None
                    )
                    if target and target in known:
                        relative = posixpath.relpath(target, posixpath.dirname(entry.name))
                        manifest["internal_links"].append({
                            "path": entry.name, "original_target": entry.linkname,
                            "target": relative,
                        })
                        entry.linkname = relative
                        entry.pax_headers.pop("linkpath", None)
                        portable.addfile(entry)
                    else:
                        manifest["external_references"].append({
                            "path": entry.name, "original_target": entry.linkname,
                            "status": "not_in_original_archive",
                        })
                    continue
                if entry.isfile():
                    payload = original.extractfile(entry).read()
                    manifest["files"].append({
                        "path": entry.name, "size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    })
                    portable.addfile(entry, io.BytesIO(payload))
                else:
                    portable.addfile(entry)
            payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
            entry = tarfile.TarInfo(ROOT + "/archive_manifest.json")
            entry.size = len(payload)
            entry.mode = 0o644
            portable.addfile(entry, io.BytesIO(payload))
    # Validate the written archive against the original regular-file hashes and
    # the link targets; do not claim that absent external datasets were recovered.
    with tarfile.open(destination, "r:gz") as written:
        members = {entry.name.rstrip("/"): entry for entry in written.getmembers()}
        for item in manifest["files"]:
            payload = written.extractfile(item["path"]).read()
            if len(payload) != item["size"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
                raise ValueError(f"Archive verification failed: {item['path']}")
        for item in manifest["internal_links"]:
            entry = members[item["path"]]
            target = posixpath.normpath(posixpath.join(posixpath.dirname(entry.name), entry.linkname))
            if entry.linkname.startswith("/") or target not in members:
                raise ValueError(f"Unresolved internal link: {entry.name}")
        if any(item["path"] in members for item in manifest["external_references"]):
            raise ValueError("External references must not be written as links")
    return {
        "schema_version": 1,
        "source_commit": source_commit,
        "original_archive": {"filename": source.name, "sha256": manifest["original_archive_sha256"], "bytes": source.stat().st_size},
        "portable_archive": {"filename": destination.name, "sha256": sha256(destination), "bytes": destination.stat().st_size},
        "regular_files": len(manifest["files"]),
        "regular_file_bytes": sum(item["size"] for item in manifest["files"]),
        "relative_internal_links": len(manifest["internal_links"]),
        "external_reference_count": len(manifest["external_references"]),
        "external_targets": sorted({item["original_target"] for item in manifest["external_references"]}),
        "dataset_complete": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()
    result = package(args.source, args.destination, args.source_commit)
    args.summary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
