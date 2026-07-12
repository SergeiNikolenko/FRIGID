#!/usr/bin/env python
"""Rerank a frozen candidate table with a trained RankLoop dual encoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from frigid.rankloop_corpus import _git_revision, sha256_file  # noqa: E402
from frigid.rankloop_inference import (  # noqa: E402
    candidate_identity_sha256,
    load_inference_candidate_frame,
    load_rankloop_checkpoint,
    score_candidate_frame,
)
from frigid.rankloop_model import (  # noqa: E402
    load_molecule_embedding_table,
    load_spectrum_embedding_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Target-blind RankLoop reranking of a frozen candidate pool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--spectrum-metadata", required=True)
    parser.add_argument("--spectrum-embeddings", required=True)
    parser.add_argument("--molecule-metadata", required=True)
    parser.add_argument("--molecule-embeddings", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in {
            "candidates": args.candidates,
            "spectrum_metadata": args.spectrum_metadata,
            "spectrum_embeddings": args.spectrum_embeddings,
            "molecule_metadata": args.molecule_metadata,
            "molecule_embeddings": args.molecule_embeddings,
            "checkpoint": args.checkpoint,
        }.items()
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} file does not exist: {path}")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = output_dir / "ranked_candidates.csv"
    manifest_path = output_dir / "run_manifest.json"
    if ranking_path.exists() or manifest_path.exists():
        raise FileExistsError(
            f"RankLoop output directory already contains result files: {output_dir}"
        )

    candidates = load_inference_candidate_frame(paths["candidates"])
    input_identity_hash = candidate_identity_sha256(candidates)
    spectrum_table = load_spectrum_embedding_table(
        paths["spectrum_metadata"], paths["spectrum_embeddings"]
    )
    molecule_table = load_molecule_embedding_table(
        paths["molecule_metadata"], paths["molecule_embeddings"]
    )
    model, checkpoint_payload = load_rankloop_checkpoint(paths["checkpoint"])
    device = resolve_device(args.device)
    ranked = score_candidate_frame(
        candidates,
        spectrum_table,
        molecule_table,
        model,
        device=device,
        batch_size=args.batch_size,
    )
    output_identity_hash = candidate_identity_sha256(ranked)
    if output_identity_hash != input_identity_hash:
        raise AssertionError("RankLoop reranking changed the frozen candidate pool.")
    ranked.to_csv(ranking_path, index=False, float_format="%.9g")

    revision, dirty = _git_revision(PROJECT_ROOT)
    manifest = {
        "schema_version": 1,
        "state": "completed",
        "repo": {"commit": revision, "dirty": dirty},
        "device": str(device),
        "parameters": vars(args),
        "model": {
            "architecture": checkpoint_payload["architecture"],
            "config": checkpoint_payload["model_config"],
            "training_input_hashes": checkpoint_payload.get("input_hashes", {}),
            "training_epoch": checkpoint_payload.get("epoch"),
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "candidate_pool": {
            "query_count": int(ranked["query_spec_name"].nunique()),
            "candidate_count": int(len(ranked)),
            "identity_sha256": input_identity_hash,
            "unchanged_after_reranking": True,
        },
        "outputs": {
            "ranked_candidates_csv": str(ranking_path),
            "ranked_candidates_sha256": sha256_file(ranking_path),
            "candidate_identity_sha256": output_identity_hash,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest["outputs"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
