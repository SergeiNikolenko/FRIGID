#!/usr/bin/env python
"""Compare two paired FRIGID benchmark runs with bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_METRICS = (
    "tanimoto_top1",
    "tanimoto_top10",
    "exact_match_top1",
    "exact_match_top10",
    "formula_success",
    "total_formula_matched",
    "total_valid",
    "total_generated",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two detailed_results.csv files on the exact same spectra.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reference", required=True, help="Reference run directory or CSV path."
    )
    parser.add_argument(
        "--candidate", required=True, help="Candidate run directory or CSV path."
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for comparison artifacts."
    )
    parser.add_argument("--reference-name", default=None)
    parser.add_argument("--candidate-name", default=None)
    parser.add_argument(
        "--fingerprint-source",
        default=None,
        help="Required when a CSV contains more than one fingerprint source.",
    )
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_results_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "detailed_results.csv"
    if not resolved.is_file():
        raise FileNotFoundError(f"Detailed results not found: {resolved}")
    return resolved


def load_results(path: Path, fingerprint_source: str | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "spec_name" not in frame.columns:
        raise ValueError(f"Missing spec_name column: {path}")

    if "fingerprint_source" in frame.columns:
        sources = sorted(frame["fingerprint_source"].dropna().unique().tolist())
        if fingerprint_source is None and len(sources) > 1:
            raise ValueError(
                f"{path} contains multiple fingerprint sources {sources}; "
                "pass --fingerprint-source."
            )
        if fingerprint_source is not None:
            frame = frame[frame["fingerprint_source"] == fingerprint_source].copy()
            if frame.empty:
                raise ValueError(
                    f"Fingerprint source {fingerprint_source!r} is absent from {path}."
                )

    if frame.empty:
        raise ValueError(f"No benchmark rows found: {path}")
    duplicates = frame.loc[frame["spec_name"].duplicated(), "spec_name"].unique()
    if duplicates.size:
        preview = ", ".join(map(str, duplicates[:5]))
        raise ValueError(f"Duplicate spec_name values in {path}: {preview}")
    return frame.reset_index(drop=True)


def add_derived_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    if "formula_success" not in enriched and "total_formula_matched" in enriched:
        enriched["formula_success"] = (enriched["total_formula_matched"] > 0).astype(
            float
        )
    return enriched


def validate_and_align(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    reference_names = reference["spec_name"].astype(str).tolist()
    candidate_names = candidate["spec_name"].astype(str).tolist()
    reference_set = set(reference_names)
    candidate_set = set(candidate_names)
    if reference_set != candidate_set:
        missing = sorted(reference_set - candidate_set)[:5]
        extra = sorted(candidate_set - reference_set)[:5]
        raise ValueError(
            "Paired subset mismatch: "
            f"reference={len(reference_set)}, candidate={len(candidate_set)}, "
            f"missing_in_candidate={missing}, extra_in_candidate={extra}"
        )

    same_order = reference_names == candidate_names
    candidate_indexed = candidate.assign(
        spec_name=candidate["spec_name"].astype(str)
    ).set_index("spec_name", drop=False)
    candidate_aligned = candidate_indexed.loc[reference_names].reset_index(drop=True)
    reference_aligned = reference.assign(
        spec_name=reference["spec_name"].astype(str)
    ).reset_index(drop=True)

    for column in ("target_smiles", "target_inchi_key", "fingerprint_source"):
        if column in reference_aligned and column in candidate_aligned:
            left = reference_aligned[column].fillna("").astype(str).to_numpy()
            right = candidate_aligned[column].fillna("").astype(str).to_numpy()
            if not np.array_equal(left, right):
                raise ValueError(f"Paired input mismatch in column {column!r}.")
    if "mist_tanimoto" in reference_aligned and "mist_tanimoto" in candidate_aligned:
        if not np.allclose(
            reference_aligned["mist_tanimoto"].to_numpy(float),
            candidate_aligned["mist_tanimoto"].to_numpy(float),
            rtol=0.0,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError("Paired input mismatch in column 'mist_tanimoto'.")
    return reference_aligned, candidate_aligned, same_order


def bootstrap_mean_interval(
    deltas: np.ndarray,
    resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    n_pairs = deltas.size
    max_index_elements = 4_000_000
    batch_size = max(1, min(512, max_index_elements // n_pairs))
    bootstrap_means = np.empty(resamples, dtype=np.float64)
    offset = 0
    while offset < resamples:
        current = min(batch_size, resamples - offset)
        indices = rng.integers(0, n_pairs, size=(current, n_pairs))
        bootstrap_means[offset : offset + current] = deltas[indices].mean(axis=1)
        offset += current
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(bootstrap_means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def hash_names(names: Iterable[str]) -> str:
    payload = "".join(f"{name}\n" for name in names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compare_runs(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    metrics: Iterable[str],
    bootstrap_resamples: int,
    confidence: float,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    reference, candidate, same_order = validate_and_align(reference, candidate)
    reference = add_derived_metrics(reference)
    candidate = add_derived_metrics(candidate)
    metric_names = list(metrics)
    missing = [
        metric
        for metric in metric_names
        if metric not in reference.columns or metric not in candidate.columns
    ]
    if missing:
        raise ValueError(f"Metrics missing from one or both runs: {missing}")

    names = reference["spec_name"].tolist()
    paired = pd.DataFrame({"spec_name": names})
    summary = {
        "schema_version": 1,
        "n_pairs": len(names),
        "same_input_order": same_order,
        "subset_sha256_ordered": hash_names(names),
        "subset_sha256_sorted": hash_names(sorted(names)),
        "bootstrap": {
            "resamples": bootstrap_resamples,
            "confidence": confidence,
            "seed": seed,
        },
        "metrics": {},
    }
    rng = np.random.default_rng(seed)
    for metric in metric_names:
        reference_values = reference[metric].to_numpy(dtype=np.float64)
        candidate_values = candidate[metric].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(reference_values).all()
            or not np.isfinite(candidate_values).all()
        ):
            raise ValueError(f"Metric {metric!r} contains non-finite values.")
        deltas = candidate_values - reference_values
        ci_low, ci_high = bootstrap_mean_interval(
            deltas,
            bootstrap_resamples,
            confidence,
            rng,
        )
        paired[f"{metric}_reference"] = reference_values
        paired[f"{metric}_candidate"] = candidate_values
        paired[f"{metric}_delta"] = deltas
        summary["metrics"][metric] = {
            "reference_mean": float(reference_values.mean()),
            "candidate_mean": float(candidate_values.mean()),
            "mean_delta": float(deltas.mean()),
            "median_delta": float(np.median(deltas)),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "wins": int((deltas > 0).sum()),
            "losses": int((deltas < 0).sum()),
            "ties": int((deltas == 0).sum()),
        }
    return summary, paired


def write_outputs(output_dir: Path, summary: dict, paired: pd.DataFrame) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    paired.to_csv(output_dir / "paired_deltas.csv", index=False)
    bootstrap = {
        "n_pairs": summary["n_pairs"],
        "bootstrap": summary["bootstrap"],
        "metrics": {
            name: {
                "mean_delta": values["mean_delta"],
                "ci_low": values["ci_low"],
                "ci_high": values["ci_high"],
            }
            for name, values in summary["metrics"].items()
        },
    }
    with (output_dir / "bootstrap_ci.json").open("w") as handle:
        json.dump(bootstrap, handle, indent=2)


def main() -> int:
    args = parse_args()
    reference_path = resolve_results_path(args.reference)
    candidate_path = resolve_results_path(args.candidate)
    reference = load_results(reference_path, args.fingerprint_source)
    candidate = load_results(candidate_path, args.fingerprint_source)
    summary, paired = compare_runs(
        reference,
        candidate,
        args.metrics,
        args.bootstrap_resamples,
        args.confidence,
        args.seed,
    )
    summary.update(
        {
            "reference": {
                "name": args.reference_name or reference_path.parent.name,
                "path": str(reference_path),
            },
            "candidate": {
                "name": args.candidate_name or candidate_path.parent.name,
                "path": str(candidate_path),
            },
            "fingerprint_source": args.fingerprint_source,
        }
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    write_outputs(output_dir, summary, paired)

    print(f"Compared {summary['n_pairs']} paired spectra")
    for metric, values in summary["metrics"].items():
        print(
            f"{metric}: delta={values['mean_delta']:.6f}, "
            f"CI=[{values['ci_low']:.6f}, {values['ci_high']:.6f}], "
            f"wins/losses/ties={values['wins']}/{values['losses']}/{values['ties']}"
        )
    print(f"Artifacts written to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
