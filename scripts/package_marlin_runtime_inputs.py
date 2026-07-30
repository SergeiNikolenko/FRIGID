#!/usr/bin/env python3
"""Build a pinned FARO runtime bundle including held-out DreaMS predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.reproduction_root.resolve()
    sources = {
        "length_audit.json": (
            root / "manifests/safe_gpt_eligibility_a7a49f6_first1000000.json"
        ),
        "nplib1_test_inchikeys.csv": root / "data/nplib1_test_inchikeys.csv",
        "tokenizer.json": root / "data/safe-gpt/tokenizer.json",
        "val/dreams_predictions.npz": root / "runs/dreams/probe/predictions.npz",
        "val/dreams_predictions.summary.json": root / "runs/dreams/probe/summary.json",
        "val/fingerprints.npz": root / "data/processed/val/fingerprints.npz",
        "val/metadata.csv": root / "data/processed/val/metadata.csv",
        "test/dreams_predictions.npz": (
            root / "runs/dreams/probe/test_predictions.npz"
        ),
        "test/dreams_predictions.summary.json": (
            root / "runs/dreams/probe/test_predictions.summary.json"
        ),
        "test/metadata.csv": root / "data/processed/test/metadata.csv",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing runtime inputs: " + ", ".join(missing))

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix=".marlin-runtime-inputs."
    ) as temporary_text:
        temporary = Path(temporary_text)
        files = []
        for relative, source in sorted(sources.items()):
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            files.append(
                {
                    "path": relative,
                    "size_bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                }
            )
        manifest = {
            "schema_version": 1,
            "kind": "MARLIN FARO runtime input bundle",
            "files": files,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        partial = temporary / output.name
        with tarfile.open(partial, "w:gz") as archive:
            archive.add(temporary / "manifest.json", arcname="manifest.json")
            for entry in files:
                archive.add(temporary / entry["path"], arcname=entry["path"])
        shutil.copyfile(partial, output)

    print(f"bundle={output}")
    print(f"sha256={sha256_file(output)}")


if __name__ == "__main__":
    main()
