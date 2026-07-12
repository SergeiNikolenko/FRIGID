#!/usr/bin/env python
"""Fuse frozen MIST and RankLoop scores without target-derived inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from frigid.rankloop_corpus import _git_revision, sha256_file  # noqa: E402
from frigid.rankloop_fusion import fuse_rankloop_scores  # noqa: E402
from frigid.rankloop_inference import (  # noqa: E402
    candidate_identity_sha256,
    load_inference_candidate_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Target-blind residual fusion of MIST and RankLoop scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--rankloop-ranked", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--normalization", choices=("zscore", "rank"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.rankloop_ranked).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"RankLoop ranking does not exist: {input_path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = output_dir / "ranked_candidates.csv"
    manifest_path = output_dir / "run_manifest.json"
    if ranking_path.exists() or manifest_path.exists():
        raise FileExistsError(f"RankLoop fusion output already exists: {output_dir}")

    frame = load_inference_candidate_frame(input_path)
    input_identity_hash = candidate_identity_sha256(frame)
    ranked = fuse_rankloop_scores(
        frame,
        alpha=args.alpha,
        normalization=args.normalization,
    )
    output_identity_hash = candidate_identity_sha256(ranked)
    ranked.to_csv(ranking_path, index=False, float_format="%.9g")

    revision, dirty = _git_revision(PROJECT_ROOT)
    manifest = {
        "schema_version": 1,
        "state": "completed",
        "repo": {"commit": revision, "dirty": dirty},
        "parameters": vars(args),
        "target_use": "none",
        "inputs": {
            "rankloop_ranked": {
                "path": str(input_path),
                "sha256": sha256_file(input_path),
            }
        },
        "candidate_pool": {
            "query_count": int(ranked["query_spec_name"].nunique()),
            "candidate_count": int(len(ranked)),
            "identity_sha256": input_identity_hash,
            "output_identity_sha256": output_identity_hash,
            "unchanged": input_identity_hash == output_identity_hash,
        },
        "outputs": {
            "ranked_candidates_csv": str(ranking_path),
            "ranked_candidates_sha256": sha256_file(ranking_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest["outputs"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
