#!/usr/bin/env python3
"""Download a large Zenodo file with verified parallel HTTP ranges."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import time
import urllib.request
from pathlib import Path


def download_part(url: str, part: Path, start: int, end: int, retries: int) -> Path:
    expected = end - start + 1
    if part.exists() and part.stat().st_size == expected:
        return part
    for attempt in range(1, retries + 1):
        temporary = part.with_suffix(".partial")
        temporary.unlink(missing_ok=True)
        request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            if temporary.stat().st_size != expected:
                raise OSError(f"range {start}-{end} returned {temporary.stat().st_size} bytes")
            temporary.replace(part)
            return part
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 30))
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--md5", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk-mib", type=int, default=32)
    parser.add_argument("--retries", type=int, default=8)
    args = parser.parse_args()

    chunk_size = args.chunk_mib * 1024 * 1024
    parts = args.output.with_suffix(args.output.suffix + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    ranges = []
    for index, start in enumerate(range(0, args.size, chunk_size)):
        end = min(start + chunk_size, args.size) - 1
        ranges.append((args.url, parts / f"{index:05d}.part", start, end, args.retries))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for completed, _ in enumerate(executor.map(lambda values: download_part(*values), ranges), start=1):
            print(f"downloaded {completed}/{len(ranges)} ranges", flush=True)

    temporary = args.output.with_suffix(args.output.suffix + ".assembling")
    digest = hashlib.md5()
    with temporary.open("wb") as output:
        for _, part, _, _, _ in ranges:
            with part.open("rb") as source:
                while block := source.read(1024 * 1024):
                    output.write(block)
                    digest.update(block)
    if temporary.stat().st_size != args.size:
        raise OSError(f"assembled size mismatch: {temporary.stat().st_size} != {args.size}")
    if digest.hexdigest() != args.md5:
        raise OSError(f"MD5 mismatch: {digest.hexdigest()} != {args.md5}")
    temporary.replace(args.output)
    for part in parts.iterdir():
        part.unlink()
    parts.rmdir()
    print(f"saved {args.output} md5={digest.hexdigest()}")


if __name__ == "__main__":
    main()
