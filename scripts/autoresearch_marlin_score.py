#!/usr/bin/env python3
"""Immutable, cache-backed molecular validation scorer for MARLIN research."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = Path(
    os.environ.get(
        "MARLIN_REPRO_ROOT",
        "/home/nikolenko/work/Projects/MARLIN_reproduction_20260717",
    )
)
DEFAULT_SHARED_ROOT = Path("/mnt/netstorage/nikolenko/marlin")
DEFAULT_METADATA = REPRO_ROOT / "data/processed/val/metadata.csv"
DEFAULT_FINGERPRINTS = (
    DEFAULT_SHARED_ROOT / "runtime-inputs-v3-0a1a6afd/val/dreams_predictions.npz"
)
DEFAULT_TOKENIZER = REPRO_ROOT / "data/safe-gpt/tokenizer.json"
DEFAULT_SPEC_MANIFEST = (
    PROJECT_ROOT / "configs/benchmarks/nplib1_v1/nplib1_val_micro32_v1.tsv"
)

SCORE_WEIGHTS = {
    "exact_top1": 0.30,
    "exact_top10": 0.20,
    "tanimoto_top1": 0.20,
    "candidate_return_rate": 0.10,
    "mass_validity": 0.10,
    "validity": 0.05,
    "uniqueness": 0.05,
}
PROTECTED_EVALUATOR_PATHS = (
    "scripts/autoresearch_marlin_score.py",
    "scripts/run_marlin_autoresearch_score.sh",
    "scripts/slurm_marlin_autoresearch_score.sbatch",
    "scripts/evaluate_marlin_nplib1.py",
    "src/marlin/benchmark_selection.py",
    "src/marlin/evaluation.py",
    "src/marlin/research_metric.py",
    "configs/benchmarks/nplib1_v1/",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--fingerprints", type=Path, default=DEFAULT_FINGERPRINTS)
    parser.add_argument("--fingerprint-key", default="probs")
    parser.add_argument("--fingerprint-threshold", type=float, default=0.95)
    parser.add_argument("--spec-manifest", type=Path, default=DEFAULT_SPEC_MANIFEST)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--shared-root", type=Path, default=DEFAULT_SHARED_ROOT)
    parser.add_argument("--seeds", default="42,314159,271828")
    parser.add_argument("--max-spectra", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--diversity-dropout", type=float, default=0.3)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def finite_metric(metrics: dict[str, Any], name: str) -> float:
    value = float(metrics.get(name, 0.0))
    return value if math.isfinite(value) else 0.0


def aggregate_seed_metrics(seed_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not seed_metrics:
        raise ValueError("at least one seed result is required")
    metric_names = tuple(SCORE_WEIGHTS) + (
        "tanimoto_top10",
        "formula_top1_all",
        "formula_top10_all",
        "internal_diversity",
        "runtime_seconds_total",
    )
    aggregate = {
        name: sum(finite_metric(metrics, name) for metrics in seed_metrics)
        / len(seed_metrics)
        for name in metric_names
    }
    aggregate["marlin_validation_score"] = sum(
        SCORE_WEIGHTS[name] * aggregate[name] for name in SCORE_WEIGHTS
    )
    from marlin.research_metric import staged_research_metric

    aggregate.update(staged_research_metric(aggregate))
    aggregate["score_weights"] = SCORE_WEIGHTS
    aggregate["seed_count"] = len(seed_metrics)
    return aggregate


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def require_clean_evaluator() -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *PROTECTED_EVALUATOR_PATHS],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            "refusing to score with a modified evaluator contract:\n" + status
        )


def main() -> None:
    args = parse_args()
    require_clean_evaluator()
    seeds = tuple(int(seed.strip()) for seed in args.seeds.split(",") if seed.strip())
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    if args.max_spectra < 1 or args.candidates < 1:
        raise ValueError("--max-spectra and --candidates must be positive")
    for path in (
        args.checkpoint,
        args.metadata,
        args.fingerprints,
        args.tokenizer,
        args.spec_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    evaluator = PROJECT_ROOT / "scripts/evaluate_marlin_nplib1.py"
    source_digest = stable_digest(
        {
            "scorer": sha256_file(Path(__file__)),
            "evaluator": sha256_file(evaluator),
            "benchmark_selection": sha256_file(
                PROJECT_ROOT / "src/marlin/benchmark_selection.py"
            ),
            "research_metric": sha256_file(
                PROJECT_ROOT / "src/marlin/research_metric.py"
            ),
        }
    )
    checkpoint_sha256 = sha256_file(args.checkpoint)
    cache_payload = {
        "schema_version": 1,
        "checkpoint_sha256": checkpoint_sha256,
        "source_digest": source_digest,
        "metadata_sha256": sha256_file(args.metadata),
        "fingerprints_sha256": sha256_file(args.fingerprints),
        "fingerprint_key": args.fingerprint_key,
        "fingerprint_threshold": args.fingerprint_threshold,
        "spec_manifest_sha256": sha256_file(args.spec_manifest),
        "tokenizer_sha256": sha256_file(args.tokenizer),
        "seeds": seeds,
        "max_spectra": args.max_spectra,
        "candidates": args.candidates,
        "temperature": args.temperature,
        "diversity_dropout": args.diversity_dropout,
        "generation_mode": "block",
        "token_selection": "multinomial",
        "mass_shell": True,
        "grammar_mask": True,
    }
    cache_key = stable_digest(cache_payload)[:20]
    cache_dir = (
        args.shared_root / "runs/autoresearch/scorer-cache" / cache_key
    )
    cached_aggregate = cache_dir / "aggregate_metrics.json"
    if cached_aggregate.is_file():
        payload = json.loads(cached_aggregate.read_text())
        payload["cache_hit"] = True
        write_json_atomic(args.output, payload)
        print(json.dumps(payload, sort_keys=True))
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    seed_metrics = []
    for seed in seeds:
        seed_dir = cache_dir / f"seed-{seed}"
        metrics_path = seed_dir / "metrics.json"
        if not metrics_path.is_file():
            command = [
                sys.executable,
                str(evaluator),
                "--checkpoint",
                str(args.checkpoint),
                "--tokenizer",
                str(args.tokenizer),
                "--metadata",
                str(args.metadata),
                "--fingerprints",
                str(args.fingerprints),
                "--fingerprint-key",
                args.fingerprint_key,
                "--threshold",
                str(args.fingerprint_threshold),
                "--spec-manifest",
                str(args.spec_manifest),
                "--lane",
                "dreams",
                "--output-dir",
                str(seed_dir),
                "--candidates",
                str(args.candidates),
                "--max-spectra",
                str(args.max_spectra),
                "--diversity-dropout",
                str(args.diversity_dropout),
                "--temperature",
                str(args.temperature),
                "--generation-mode",
                "block",
                "--ppm-tolerance",
                "10.0",
                "--seed",
                str(seed),
                "--sample-tokens",
            ]
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        seed_metrics.append(json.loads(metrics_path.read_text()))

    payload = {
        **aggregate_seed_metrics(seed_metrics),
        "schema_version": 1,
        "cache_hit": False,
        "cache_key": cache_key,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "validation_rows_per_seed": args.max_spectra,
        "candidates_per_row": args.candidates,
        "seeds": list(seeds),
        "selection_split": "held-out validation",
        "locked_test_used": False,
        "fingerprint_contract": {
            "source": "spectrum-derived DreaMS prediction",
            "key": args.fingerprint_key,
            "threshold": args.fingerprint_threshold,
            "oracle_used": False,
        },
        "spec_manifest": str(args.spec_manifest.resolve()),
        "paper_recipe_guards": {
            "block_width": "read from checkpoint",
            "symmetric_fingerprint_noise": True,
            "candidate_conditioning_diversity": args.diversity_dropout,
            "mass_shell_decoding": True,
            "token_selection": "multinomial fixed for all candidates",
        },
        "seed_metrics": seed_metrics,
        "runtime": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": os.uname().nodename,
        },
    }
    write_json_atomic(cached_aggregate, payload)
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
