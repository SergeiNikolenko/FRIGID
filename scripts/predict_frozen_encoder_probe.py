#!/usr/bin/env python3
"""Apply a trained frozen-encoder fingerprint probe to an embedding bundle."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from frigid.encoder_benchmark import sha256_file  # noqa: E402
from frigid.frozen_probe import load_embedding_bundle  # noqa: E402
from train_frozen_encoder_probe import FrozenFingerprintProbe  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    bundle = load_embedding_bundle(args.bundle, fingerprint_bits=4096)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = FrozenFingerprintProbe(
        input_dim=int(checkpoint["input_dim"]),
        fingerprint_bits=int(checkpoint["fingerprint_bits"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(bundle.embeddings)),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    predictions = []
    started = time.perf_counter()
    with torch.inference_mode():
        for (embeddings,) in loader:
            logits = model(embeddings.to(device=device, dtype=torch.float32))
            predictions.append(torch.sigmoid(logits).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        probs=np.vstack(predictions).astype(np.float32, copy=False),
        spectrum_ids=bundle.spectrum_ids,
        inference_seconds=np.full(
            len(bundle.spectrum_ids), elapsed / len(bundle.spectrum_ids), dtype=np.float64
        ),
    )
    summary = {
        "rows": len(bundle.spectrum_ids),
        "device": str(device),
        "elapsed_seconds": elapsed,
        "bundle_sha256": sha256_file(args.bundle),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "output_sha256": sha256_file(args.output),
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
