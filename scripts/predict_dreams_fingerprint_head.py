#!/usr/bin/env python
"""Run a trained DreaMS fingerprint head on precomputed embeddings."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from frigid.dreams_fingerprint_head import FingerprintHead, load_npz_array  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict Morgan fingerprint probabilities from DreaMS embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embeddings-npz", required=True)
    parser.add_argument("--embeddings-key", default="embeddings")
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location="cpu")
    threshold = args.threshold if args.threshold is not None else payload.get("threshold", 0.5)
    embeddings = load_npz_array(args.embeddings_npz, args.embeddings_key).astype(np.float32)

    model = FingerprintHead(
        input_dim=payload["input_dim"],
        fingerprint_bits=payload["fingerprint_bits"],
        hidden_dim=payload["hidden_dim"],
        dropout=payload["dropout"],
    )
    model.load_state_dict(payload["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    loader = DataLoader(TensorDataset(torch.as_tensor(embeddings)), batch_size=args.batch_size)
    probs = []
    with torch.no_grad():
        for (batch_embeddings,) in loader:
            logits = model(batch_embeddings.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    probs_np = np.vstack(probs).astype(np.float32)
    binary_np = (probs_np >= threshold).astype(np.float32)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "fingerprints.npz"
    metadata_out = output_dir / "metadata.csv"
    summary_path = output_dir / "summary.json"

    np.savez_compressed(predictions_path, probs=probs_np, binary=binary_np)
    metadata_out.write_text(Path(args.metadata_csv).read_text())
    summary_path.write_text(
        json.dumps(
            {
                "rows": int(len(probs_np)),
                "fingerprint_bits": int(probs_np.shape[1]),
                "threshold": float(threshold),
                "checkpoint": args.checkpoint,
                "embeddings_npz": args.embeddings_npz,
                "metadata_csv": str(metadata_out),
                "fingerprints_npz": str(predictions_path),
            },
            indent=2,
        )
    )

    print(f"Saved predictions: {predictions_path}")
    print(f"Saved metadata: {metadata_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
