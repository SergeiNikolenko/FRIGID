#!/usr/bin/env python
"""Evaluate RankLoop only after target-blind candidate scores are frozen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
SCRIPTS_PATH = PROJECT_ROOT / "scripts"
for path in (SRC_PATH, SCRIPTS_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compare_paired_benchmark_runs import compare_runs, write_outputs  # noqa: E402
from frigid.rankloop_corpus import _git_revision, sha256_file  # noqa: E402
from frigid.rankloop_evaluation import (  # noqa: E402
    evaluate_ranked_candidates,
    load_ranked_candidate_frame,
    load_target_metadata,
    validate_identical_candidate_pool,
)


METRICS = (
    "tanimoto_top1",
    "tanimoto_top10",
    "exact_match_top1",
    "exact_match_top10",
    "candidate_recall_exact",
    "candidate_oracle_tanimoto",
    "ranking_regret_top1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc paired evaluation of a frozen RankLoop reranking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reference-ranked", required=True)
    parser.add_argument("--rankloop-ranked", required=True)
    parser.add_argument("--target-metadata", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--fingerprint-bits", type=int, default=4096)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        "reference_ranked": Path(args.reference_ranked).expanduser().resolve(),
        "rankloop_ranked": Path(args.rankloop_ranked).expanduser().resolve(),
        "target_metadata": Path(args.target_metadata).expanduser().resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Evaluation output directory is not empty: {output_dir}")

    reference_ranked = load_ranked_candidate_frame(
        paths["reference_ranked"], rank_column="rank"
    )
    rankloop_ranked = load_ranked_candidate_frame(
        paths["rankloop_ranked"], rank_column="rankloop_rank"
    )
    candidate_identity_hash = validate_identical_candidate_pool(
        reference_ranked, rankloop_ranked
    )
    targets = load_target_metadata(paths["target_metadata"])
    reference_detailed = evaluate_ranked_candidates(
        reference_ranked,
        targets,
        rank_column="rank",
        method_name="frozen_reference",
        top_k=args.top_k,
        fingerprint_bits=args.fingerprint_bits,
        fingerprint_radius=args.fingerprint_radius,
    )
    rankloop_detailed = evaluate_ranked_candidates(
        rankloop_ranked,
        targets,
        rank_column="rankloop_rank",
        method_name="rankloop",
        top_k=args.top_k,
        fingerprint_bits=args.fingerprint_bits,
        fingerprint_radius=args.fingerprint_radius,
    )
    reference_path = output_dir / "reference_detailed_results.csv"
    rankloop_path = output_dir / "rankloop_detailed_results.csv"
    reference_detailed.to_csv(reference_path, index=False)
    rankloop_detailed.to_csv(rankloop_path, index=False)

    summary, paired = compare_runs(
        reference_detailed,
        rankloop_detailed,
        METRICS,
        args.bootstrap_resamples,
        args.confidence,
        args.seed,
        bootstrap_unit="molecule",
        cluster_column="target_inchi_key",
    )
    summary.update(
        {
            "reference": "frozen_reference",
            "candidate": "rankloop",
            "candidate_identity_sha256": candidate_identity_hash,
            "target_use": "metrics_only_after_target_blind_ranking",
        }
    )
    comparison_dir = output_dir / "paired_comparison"
    write_outputs(comparison_dir, summary, paired)

    revision, dirty = _git_revision(PROJECT_ROOT)
    manifest = {
        "schema_version": 1,
        "state": "completed",
        "repo": {"commit": revision, "dirty": dirty},
        "parameters": vars(args),
        "target_use": "metrics_only_after_target_blind_ranking",
        "candidate_pool": {
            "query_count": int(reference_ranked["query_spec_name"].nunique()),
            "candidate_count": int(len(reference_ranked)),
            "identity_sha256": candidate_identity_hash,
            "unchanged": True,
        },
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
        "outputs": {
            "reference_detailed_results": str(reference_path),
            "reference_detailed_results_sha256": sha256_file(reference_path),
            "rankloop_detailed_results": str(rankloop_path),
            "rankloop_detailed_results_sha256": sha256_file(rankloop_path),
            "comparison_summary": str(comparison_dir / "comparison_summary.json"),
            "comparison_summary_sha256": sha256_file(
                comparison_dir / "comparison_summary.json"
            ),
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
