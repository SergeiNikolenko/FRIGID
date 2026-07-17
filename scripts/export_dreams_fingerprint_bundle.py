#!/usr/bin/env python3
"""Export official DreaMS embeddings in the shared frozen-probe bundle format."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mgf", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--fingerprints", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--logger-path")
    args = parser.parse_args()

    from dreams.api import dreams_embeddings

    metadata = pd.read_csv(args.metadata)
    with np.load(args.fingerprints, allow_pickle=False) as arrays:
        targets = arrays["ground_truth"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    embeddings = np.asarray(
        dreams_embeddings(
            args.mgf,
            batch_size=args.batch_size,
            logger_pth=args.logger_path or str(args.output.with_suffix(".log")),
            store_embs=False,
        ),
        dtype=np.float32,
    )
    elapsed = time.perf_counter() - started
    if len(metadata) != len(targets) or len(metadata) != len(embeddings):
        raise ValueError(
            f"row mismatch: metadata={len(metadata)}, targets={len(targets)}, embeddings={len(embeddings)}"
        )
    np.savez_compressed(
        args.output,
        embeddings=embeddings,
        ground_truth=targets.astype(np.uint8),
        spectrum_ids=metadata["spec_name"].astype(str).to_numpy(),
        inchikeys=metadata["inchikey"].astype(str).to_numpy(),
        inference_seconds=np.full(len(metadata), elapsed / len(metadata), dtype=np.float64),
    )
    summary = {
        "rows": len(metadata),
        "embedding_dim": embeddings.shape[1],
        "elapsed_seconds": elapsed,
        "output": str(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
