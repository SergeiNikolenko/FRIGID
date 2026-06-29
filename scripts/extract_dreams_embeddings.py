#!/usr/bin/env python
"""Extract DreaMS embeddings for an MGF file.

This wrapper is intentionally thin because DreaMS is an external runtime with
its own dependencies. It supports the public ``dreams.api.dreams_embeddings``
entry point used by the upstream project. Run it in a DreaMS-capable
environment, then feed the resulting NPZ to ``train_dreams_fingerprint_head.py``.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract DreaMS embeddings from an MGF file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mgf", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--logger-path",
        default=None,
        help="Optional DreaMS progress log path. Defaults to <output-dir>/dreams_embeddings.log.",
    )
    parser.add_argument("--array-key", default="embeddings")
    return parser.parse_args()


def _load_dreams_embeddings_fn():
    try:
        from dreams.api import dreams_embeddings
        return dreams_embeddings
    except Exception as exc:
        raise RuntimeError(
            "Could not import dreams.api.dreams_embeddings. Install DreaMS in "
            "the active environment, for example from https://github.com/pluskal-lab/DreaMS."
        ) from exc


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "dreams_embeddings.npz"
    summary_path = output_dir / "summary.json"
    logger_path = (
        Path(args.logger_path) if args.logger_path else output_dir / "dreams_embeddings.log"
    )

    dreams_embeddings = _load_dreams_embeddings_fn()

    embeddings = dreams_embeddings(
        args.mgf,
        batch_size=args.batch_size,
        logger_pth=str(logger_path),
        store_embs=False,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected DreaMS embeddings to be 2D, got {embeddings.shape}")

    np.savez_compressed(embeddings_path, **{args.array_key: embeddings})
    summary_path.write_text(
        json.dumps(
            {
                "mgf": args.mgf,
                "rows": int(embeddings.shape[0]),
                "embedding_dim": int(embeddings.shape[1]),
                "embeddings_npz": str(embeddings_path),
                "array_key": args.array_key,
                "logger_path": str(logger_path),
            },
            indent=2,
        )
    )
    print(f"Saved embeddings: {embeddings_path}")
    print(f"Shape: {embeddings.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
