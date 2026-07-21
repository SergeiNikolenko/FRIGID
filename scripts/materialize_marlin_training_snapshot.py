#!/usr/bin/env python3
"""Materialize and verify the pinned SAFE-GPT training parquet snapshot."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from marlin.dataset import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_attempts < 1:
        raise ValueError("max attempts must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(
        args.dataset,
        revision=args.revision,
        files_metadata=True,
    )
    if info.sha != args.revision:
        raise ValueError(f"resolved dataset revision {info.sha} != {args.revision}")
    siblings = sorted(
        (
            sibling
            for sibling in info.siblings
            if sibling.rfilename.startswith("data/train/")
            and sibling.rfilename.endswith(".parquet")
        ),
        key=lambda sibling: sibling.rfilename,
    )
    if not siblings or any(sibling.lfs is None for sibling in siblings):
        raise ValueError("pinned dataset has missing parquet LFS metadata")

    records = []
    for index, sibling in enumerate(siblings, start=1):
        destination = output / sibling.rfilename
        expected_sha = sibling.lfs.sha256
        for attempt in range(1, args.max_attempts + 1):
            try:
                hf_hub_download(
                    repo_id=args.dataset,
                    repo_type="dataset",
                    revision=args.revision,
                    filename=sibling.rfilename,
                    local_dir=output,
                )
                observed_sha = sha256_file(destination)
                if destination.stat().st_size != sibling.size:
                    raise ValueError(f"size mismatch for {sibling.rfilename}")
                if observed_sha != expected_sha:
                    raise ValueError(f"SHA-256 mismatch for {sibling.rfilename}")
                break
            except Exception:
                if attempt == args.max_attempts:
                    raise
                delay = min(300, 10 * 2 ** (attempt - 1))
                print(
                    f"retrying {sibling.rfilename} in {delay}s "
                    f"(attempt {attempt}/{args.max_attempts})",
                    flush=True,
                )
                time.sleep(delay)
        records.append(
            {
                "path": sibling.rfilename,
                "size_bytes": sibling.size,
                "sha256": expected_sha,
            }
        )
        print(f"verified {index}/{len(siblings)} {sibling.rfilename}", flush=True)

    manifest = {
        "schema_version": 1,
        "kind": "MARLIN offline SAFE-GPT training snapshot",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "revision": args.revision,
        "snapshot_root": str(output),
        "files": records,
        "file_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "source": "Hugging Face pinned dataset revision and LFS SHA-256 metadata",
        "hostname": os.uname().nodename,
    }
    temporary = output / "manifest.json.tmp"
    final = output / "manifest.json"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(final)
    print(f"manifest={final} sha256={sha256_file(final)}")


if __name__ == "__main__":
    main()
