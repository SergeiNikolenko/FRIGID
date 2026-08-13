#!/usr/bin/env python
"""Price the corpus stream against what a training step consumes.

A data pipeline is only "fast enough" relative to a number, so this measures
both sides: molecules per second out of the stream (per worker and with a real
``DataLoader``), and molecules per second consumed by the optimiser at the given
global batch size and measured step rate.

    env -u LD_PRELOAD PYTHONPATH=src .venv/bin/python \
        scripts/audit_corpus_stream_throughput.py --molecules 4000 --workers 8
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from dlm.utils.utils_data import get_tokenizer
from marlin.corpus_stream import FP2MOL_SNAPSHOT, Fp2MolStream, corpus_row_groups
from marlin.encoder_error_model import EncoderErrorModel
from marlin.training import MarlinCollator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=FP2MOL_SNAPSHOT)
    parser.add_argument("--molecules", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--global-batch", type=int, default=256)
    parser.add_argument(
        "--steps-per-second",
        type=float,
        default=0.217,
        help="measured optimiser step rate of the adaptation run",
    )
    parser.add_argument("--exclude-inchikeys", type=Path, default=Path("data/exclude_inchikeys.csv"))
    parser.add_argument("--error-model", type=Path, default=None)
    arguments = parser.parse_args()

    tokenizer = get_tokenizer(None)
    refs = corpus_row_groups(arguments.snapshot)
    print(f"corpus: {len(refs)} row groups under {arguments.snapshot}")

    exclude = arguments.exclude_inchikeys if arguments.exclude_inchikeys.exists() else None

    def build_stream(seed: int = 0, limit: int | None = None):
        return Fp2MolStream(
            snapshot=arguments.snapshot,
            tokenizer=tokenizer,
            exclude_inchikeys=exclude,
            seed=seed,
            limit=limit,
            row_groups=refs,
        )

    # --- single-worker molecule rate ---
    stream = build_stream(limit=arguments.molecules)
    started = time.time()
    count = sum(1 for _ in stream)
    single = count / (time.time() - started)
    print(f"single worker: {count} molecules at {single:,.0f} mol/s/core; rejections={stream.rejections}")

    # --- DataLoader, collated into real training batches ---
    collator = MarlinCollator(tokenizer, max_length=256, exclude_inchikeys=exclude)
    loader = torch.utils.data.DataLoader(
        build_stream(seed=1),
        batch_size=arguments.batch_size,
        num_workers=arguments.workers,
        collate_fn=collator,
        persistent_workers=False,
    )
    wanted = max(arguments.molecules // arguments.batch_size, 4)
    started = time.time()
    seen = 0
    for index, batch in enumerate(loader):
        seen += int(batch["input_ids"].shape[0])
        if index + 1 >= wanted:
            break
    elapsed = time.time() - started
    loader_rate = seen / elapsed
    print(
        f"DataLoader({arguments.workers} workers, batch {arguments.batch_size}): "
        f"{seen} molecules in {elapsed:.1f}s = {loader_rate:,.0f} mol/s"
    )

    demand = arguments.global_batch * arguments.steps_per_second
    print(
        f"training demand: global batch {arguments.global_batch} x "
        f"{arguments.steps_per_second} steps/s = {demand:,.1f} mol/s"
    )
    print(f"headroom: single worker {single / demand:.1f}x, loader {loader_rate / demand:.1f}x")

    if arguments.error_model:
        model = EncoderErrorModel.load(arguments.error_model)
        fingerprints = batch["fingerprint"]
        torch.manual_seed(0)
        started = time.time()
        for _ in range(200):
            model.corrupt(fingerprints)
        per_batch = (time.time() - started) / 200
        print(
            f"corruption: {per_batch * 1e3:.3f} ms per batch of "
            f"{fingerprints.shape[0]} on CPU = {fingerprints.shape[0] / per_batch:,.0f} mol/s"
        )


if __name__ == "__main__":
    main()
