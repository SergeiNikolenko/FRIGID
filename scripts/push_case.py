#!/usr/bin/env python3
"""Publish a recorded decode to the viewer, so a case can be looked at anywhere.

The payload is whatever ``scripts/build_attempt_fan.py`` wrote. It is uploaded
straight to the blob store the site reads, which is what keeps a 20 MB case out
of the 4.5 MB request-body limit of a serverless function.

The store token is read from ``MARLIN_VIEWER_BLOB_TOKEN`` (keep it out of the
repository; ``~/.config/marlin-viewer.env`` is a good home).

Usage:
  MARLIN_VIEWER_BLOB_TOKEN=... python scripts/push_case.py \
      docs/attempt-fan/attempt_fan.json --id clean-panel-benzaldehyde
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

SITE = os.environ.get("MARLIN_VIEWER_SITE", "https://decodescope.vercel.app")
BLOB = "https://blob.vercel-storage.com"
ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path)
    parser.add_argument("--id", required=True)
    parser.add_argument("--title", help="shown on the case card; defaults to the spectra")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get("MARLIN_VIEWER_BLOB_TOKEN")
    if not token:
        print("MARLIN_VIEWER_BLOB_TOKEN is not set", file=sys.stderr)
        return 2
    if not ID.match(args.id):
        print("--id must match [a-z0-9][a-z0-9._-]{0,63}", file=sys.stderr)
        return 2

    payload = json.loads(args.payload.read_text())
    if "spectra" not in payload:
        print("payload has no 'spectra': is this a build_attempt_fan.py output?", file=sys.stderr)
        return 2
    if args.title:
        payload["title"] = args.title
    body = json.dumps(payload, separators=(",", ":")).encode()

    request = urllib.request.Request(
        f"{BLOB}/cases/{args.id}.json",
        data=body,
        method="PUT",
        headers={
            "authorization": f"Bearer {token}",
            "x-api-version": "7",
            "x-content-type": "application/json",
            "x-add-random-suffix": "0",
            "x-allow-overwrite": "1",
        },
    )
    with urllib.request.urlopen(request) as response:
        result = json.load(response)

    spectra = payload["spectra"]
    attempts = sum(len(spectrum.get("attempts", [])) for spectrum in spectra)
    print(f"pushed {len(spectra)} spectra, {attempts} attempts, {len(body) / 1e6:.1f} MB")
    print(f"  data : {result['url']}")
    print(f"  view : {SITE}/c/{args.id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
