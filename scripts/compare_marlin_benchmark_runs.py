#!/usr/bin/env python3
"""Compare paired MARLIN evaluations with molecule-cluster bootstrap CIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from marlin.benchmark_selection import hash_spec_names


DEFAULT_METRICS = (
    "exact_top1",
    "exact_top10",
    "candidate_returned",
    "formula_top1",
    "formula_top10",
    "tanimoto_top1",
    "tanimoto_top10",
    "validity",
    "mass_validity",
    "uniqueness",
)
INVARIANT_SETTINGS = (
    "lane",
    "fingerprint_key",
    "fingerprint_threshold",
    "candidates",
    "diversity_dropout",
    "temperature",
    "generation_mode",
    "ppm_tolerance",
    "valence_slack",
    "eos_boost",
    "grammar_mask",
    "mass_shell_constraint",
    "safe_decode_fix",
    "token_selection",
    "seed",
    "ordered_spec_names_sha256",
)


def resolve_run(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_file() and resolved.name == "predictions.jsonl":
        return resolved.parent
    if not (resolved / "predictions.jsonl").is_file():
        raise FileNotFoundError(f"MARLIN predictions not found: {resolved}")
    if not (resolved / "run_signature.json").is_file():
        raise FileNotFoundError(f"MARLIN run signature not found: {resolved}")
    return resolved


def load_predictions(run_dir: Path) -> pd.DataFrame:
    rows = [
        json.loads(line)
        for line in (run_dir / "predictions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"No prediction rows found: {run_dir}")
    frame = pd.DataFrame(rows)
    required = {"spec_name", "target_inchikey_first_block", "neutral_mass"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Predictions lack columns {sorted(missing)}: {run_dir}")
    frame["spec_name"] = frame["spec_name"].astype(str)
    if frame["spec_name"].duplicated().any():
        raise ValueError(f"Predictions contain duplicate spec_name values: {run_dir}")
    return frame


def validate_signatures(reference_dir: Path, candidate_dir: Path) -> dict:
    reference = json.loads((reference_dir / "run_signature.json").read_text())
    candidate = json.loads((candidate_dir / "run_signature.json").read_text())
    left = reference.get("settings", {})
    right = candidate.get("settings", {})
    required = {"ordered_spec_names_sha256", "seed", "candidates"}
    if not required.issubset(left) or not required.issubset(right):
        raise ValueError(
            "Run signatures predate the paired validation contract; rerun both "
            "evaluations with --spec-manifest"
        )
    mismatches = {
        key: {"reference": left.get(key), "candidate": right.get(key)}
        for key in INVARIANT_SETTINGS
        if left.get(key) != right.get(key)
    }
    if mismatches:
        raise ValueError(f"Paired run settings mismatch: {mismatches}")
    return {
        "reference_git_commit": reference.get("git_commit"),
        "candidate_git_commit": candidate.get("git_commit"),
        "invariant_settings": {key: left.get(key) for key in INVARIANT_SETTINGS},
    }


def align(
    reference: pd.DataFrame, candidate: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    reference_names = reference["spec_name"].tolist()
    candidate_names = candidate["spec_name"].tolist()
    if set(reference_names) != set(candidate_names):
        raise ValueError(
            "Paired spectrum mismatch: "
            f"missing={sorted(set(reference_names) - set(candidate_names))[:5]}, "
            f"extra={sorted(set(candidate_names) - set(reference_names))[:5]}"
        )
    same_order = reference_names == candidate_names
    candidate = candidate.set_index("spec_name", drop=False).loc[
        reference_names
    ].reset_index(drop=True)
    reference = reference.reset_index(drop=True)
    for column in (
        "target_inchikey_first_block",
        "target_smiles",
        "neutral_mass",
        "lane",
    ):
        if column not in reference or column not in candidate:
            continue
        left = reference[column].fillna("").astype(str).to_numpy()
        right = candidate[column].fillna("").astype(str).to_numpy()
        if not np.array_equal(left, right):
            raise ValueError(f"Paired input mismatch in {column!r}")
    return reference, candidate, same_order


def cluster_bootstrap_interval(
    deltas: np.ndarray,
    clusters: np.ndarray,
    resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive")
    if not 0 < confidence < 1:
        raise ValueError("--confidence must be between zero and one")
    unique, inverse = np.unique(clusters.astype(str), return_inverse=True)
    if len(unique) < 2:
        raise ValueError("Molecule bootstrap requires at least two clusters")
    sums = np.bincount(inverse, weights=deltas)
    counts = np.bincount(inverse)
    means = np.empty(resamples)
    batch_size = max(1, min(512, 4_000_000 // len(unique)))
    for start in range(0, resamples, batch_size):
        current = min(batch_size, resamples - start)
        indices = rng.integers(0, len(unique), size=(current, len(unique)))
        means[start : start + current] = (
            sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
        )
    alpha = (1 - confidence) / 2
    return tuple(float(value) for value in np.quantile(means, [alpha, 1 - alpha]))


def compare_runs(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    metrics: Iterable[str],
    resamples: int,
    confidence: float,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    reference, candidate, same_order = align(reference, candidate)
    names = reference["spec_name"].tolist()
    clusters = reference["target_inchikey_first_block"].astype(str).to_numpy()
    paired = pd.DataFrame(
        {"spec_name": names, "bootstrap_cluster": clusters}
    )
    summary: dict = {
        "schema_version": 1,
        "n_pairs": len(names),
        "n_molecule_clusters": int(np.unique(clusters).size),
        "same_input_order": same_order,
        "ordered_spec_names_sha256": hash_spec_names(names),
        "bootstrap": {
            "unit": "molecule_connectivity",
            "resamples": resamples,
            "confidence": confidence,
            "seed": seed,
        },
        "metrics": {},
    }
    rng = np.random.default_rng(seed)
    for metric in metrics:
        if metric not in reference or metric not in candidate:
            raise ValueError(f"Metric absent from one or both runs: {metric}")
        left = reference[metric].astype(float).to_numpy()
        right = candidate[metric].astype(float).to_numpy()
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"Metric contains non-finite values: {metric}")
        delta = right - left
        low, high = cluster_bootstrap_interval(
            delta, clusters, resamples, confidence, rng
        )
        paired[f"{metric}_reference"] = left
        paired[f"{metric}_candidate"] = right
        paired[f"{metric}_delta"] = delta
        summary["metrics"][metric] = {
            "reference_mean": float(left.mean()),
            "candidate_mean": float(right.mean()),
            "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "ci_low": low,
            "ci_high": high,
            "wins": int((delta > 0).sum()),
            "losses": int((delta < 0).sum()),
            "ties": int((delta == 0).sum()),
        }
    return summary, paired


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference_dir = resolve_run(args.reference)
    candidate_dir = resolve_run(args.candidate)
    signature_check = validate_signatures(reference_dir, candidate_dir)
    summary, paired = compare_runs(
        load_predictions(reference_dir),
        load_predictions(candidate_dir),
        args.metrics,
        args.bootstrap_resamples,
        args.confidence,
        args.seed,
    )
    summary["paired_contract"] = signature_check
    summary["reference_dir"] = str(reference_dir)
    summary["candidate_dir"] = str(candidate_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    paired.to_csv(args.output_dir / "paired_deltas.csv", index=False)
    (args.output_dir / "bootstrap_ci.json").write_text(
        json.dumps(
            {
                "n_pairs": summary["n_pairs"],
                "bootstrap": summary["bootstrap"],
                "metrics": {
                    key: {
                        "mean_delta": value["mean_delta"],
                        "ci_low": value["ci_low"],
                        "ci_high": value["ci_high"],
                    }
                    for key, value in summary["metrics"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
