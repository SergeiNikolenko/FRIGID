#!/usr/bin/env python
"""Fuse frozen FRIGID ranking with target-blind forward-spectrum scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from frigid.forward_consistency import fuse_forward_consistency_scores  # noqa: E402
from frigid.rankloop_corpus import _git_revision, sha256_file  # noqa: E402
from frigid.rankloop_inference import (  # noqa: E402
    candidate_identity_sha256,
    load_inference_candidate_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Target-blind fusion of frozen and forward-spectrum scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--forward-scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--normalization", choices=("zscore", "rank"), required=True)
    parser.add_argument("--mode", choices=("blend", "contradiction"), required=True)
    parser.add_argument("--contradiction-quantile", type=float, default=0.25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.forward_scores).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Forward score table does not exist: {input_path}")
    if output_dir.exists():
        raise FileExistsError(f"Forward fusion output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    frame = load_inference_candidate_frame(input_path)
    input_identity = candidate_identity_sha256(frame)
    ranked = fuse_forward_consistency_scores(
        frame,
        alpha=args.alpha,
        normalization=args.normalization,
        mode=args.mode,
        contradiction_quantile=args.contradiction_quantile,
    )
    output_identity = candidate_identity_sha256(ranked)
    ranking_path = output_dir / "ranked_candidates.csv"
    ranked.to_csv(ranking_path, index=False, float_format="%.9g")

    revision, dirty = _git_revision(PROJECT_ROOT)
    manifest = {
        "schema_version": 1,
        "state": "completed",
        "repo": {"commit": revision, "dirty": dirty},
        "parameters": vars(args),
        "target_use": "none",
        "inputs": {"forward_scores": {"path": str(input_path), "sha256": sha256_file(input_path)}},
        "candidate_pool": {
            "query_count": int(ranked["query_spec_name"].nunique()),
            "candidate_count": len(ranked),
            "identity_sha256": input_identity,
            "output_identity_sha256": output_identity,
            "unchanged": input_identity == output_identity,
        },
        "fallback_query_count": int(
            ranked.groupby("query_spec_name")["forward_fallback"].first().sum()
        ),
        "outputs": {
            "ranked_candidates": {"path": str(ranking_path), "sha256": sha256_file(ranking_path)}
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest["outputs"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
