#!/usr/bin/env python
"""Run a unified, ID-safe benchmark for MS/MS fingerprint encoders."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from frigid.encoder_benchmark import (  # noqa: E402
    aggregate_metrics,
    compute_per_spectrum_metrics,
    load_prediction_bundle,
    load_reference_bundle,
    load_reference_predictions,
    load_training_identifiers,
    paired_bootstrap_mean_ci,
    sha256_file,
    structure_identifiers,
    training_overlap,
)


MODEL_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def parse_named_values(values: list[str], *, kind: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{kind} must use NAME=VALUE syntax, got {value!r}")
        name, payload = value.split("=", maxsplit=1)
        if not MODEL_NAME.fullmatch(name):
            raise ValueError(f"Invalid model name {name!r}")
        if not payload:
            raise ValueError(f"{kind} for {name!r} is empty")
        if name in parsed:
            raise ValueError(f"Duplicate {kind} for model {name!r}")
        parsed[name] = payload
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare spectrum-to-fingerprint encoders on one explicit benchmark manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reference-metadata", required=True)
    parser.add_argument("--reference-fingerprints", required=True)
    parser.add_argument("--target-key", default="ground_truth")
    parser.add_argument("--id-column", default="spec_name")
    parser.add_argument(
        "--reference-model",
        action="append",
        default=[],
        metavar="NAME=ARRAY_KEY",
        help="Evaluate a row-aligned prediction array from --reference-fingerprints.",
    )
    parser.add_argument(
        "--prediction",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Evaluate a standard NPZ containing spectrum_ids and probs.",
    )
    parser.add_argument(
        "--prediction-metadata",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Companion metadata for a historical prediction NPZ without embedded IDs.",
    )
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Model-specific binary threshold fixed before evaluation.",
    )
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=None,
        help="Optional fallback for smoke tests; final benchmarks should set every threshold explicitly.",
    )
    parser.add_argument("--expected-bits", type=int, default=4096)
    parser.add_argument("--baseline", required=True, help="Model used for paired deltas.")
    parser.add_argument("--minimum-gain", type=float, default=0.005)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument(
        "--code-revision",
        default=None,
        help="Exact benchmark code commit or immutable revision for copied runners.",
    )
    parser.add_argument(
        "--stratify-column",
        action="append",
        default=[],
        help="Metadata column for categorical error analysis; may be repeated.",
    )
    parser.add_argument(
        "--training-identifiers",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Optional CSV/TXT training structure identifiers for leakage audit.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reference_models = parse_named_values(args.reference_model, kind="reference model")
    prediction_paths = parse_named_values(args.prediction, kind="prediction")
    prediction_metadata_paths = parse_named_values(
        args.prediction_metadata, kind="prediction metadata"
    )
    threshold_values = parse_named_values(args.threshold, kind="threshold")
    training_paths = parse_named_values(args.training_identifiers, kind="training identifiers")

    if args.expected_bits <= 0:
        raise ValueError(f"Expected fingerprint bits must be positive, got {args.expected_bits}")
    if not np.isfinite(args.minimum_gain) or args.minimum_gain < 0.0:
        raise ValueError(f"Minimum gain must be non-negative, got {args.minimum_gain}")
    if args.bootstrap_samples <= 0:
        raise ValueError(
            f"Bootstrap samples must be positive, got {args.bootstrap_samples}"
        )

    overlap = set(reference_models) & set(prediction_paths)
    if overlap:
        raise ValueError(f"Models cannot be both reference and external predictions: {sorted(overlap)}")
    model_names = list(reference_models) + list(prediction_paths)
    if not model_names:
        raise ValueError("At least one --reference-model or --prediction is required")
    if args.baseline not in model_names:
        raise ValueError(f"Baseline {args.baseline!r} is not one of the evaluated models")
    unknown_thresholds = set(threshold_values) - set(model_names)
    if unknown_thresholds:
        raise ValueError(f"Thresholds were supplied for unknown models: {sorted(unknown_thresholds)}")
    unknown_prediction_metadata = set(prediction_metadata_paths) - set(prediction_paths)
    if unknown_prediction_metadata:
        raise ValueError(
            "Prediction metadata was supplied without a matching external prediction for "
            f"{sorted(unknown_prediction_metadata)}"
        )
    unknown_training = set(training_paths) - set(model_names)
    if unknown_training:
        raise ValueError(
            f"Training identifiers were supplied for unknown models: {sorted(unknown_training)}"
        )

    missing_thresholds = set(model_names) - set(threshold_values)
    if missing_thresholds and args.default_threshold is None:
        raise ValueError(
            "Every model needs an explicit validation-frozen --threshold; missing for "
            f"{sorted(missing_thresholds)}"
        )
    thresholds = {
        name: float(
            threshold_values[name] if name in threshold_values else args.default_threshold
        )
        for name in model_names
    }
    reference = load_reference_bundle(
        args.reference_metadata,
        args.reference_fingerprints,
        target_key=args.target_key,
        id_column=args.id_column,
    )
    if reference.fingerprint_bits != args.expected_bits:
        raise ValueError(
            f"Reference uses {reference.fingerprint_bits} fingerprint bits; "
            f"the locked benchmark expects {args.expected_bits}"
        )
    molecule_ids = structure_identifiers(reference.metadata)
    missing_strata = [column for column in args.stratify_column if column not in reference.metadata]
    if missing_strata:
        raise ValueError(f"Requested stratification columns are missing: {missing_strata}")

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    per_model: dict[str, pd.DataFrame] = {}
    aggregates: dict[str, dict] = {}
    provenance: dict[str, dict] = {}

    for name in model_names:
        if name in reference_models:
            predictions = load_reference_predictions(
                args.reference_fingerprints,
                reference_models[name],
                reference,
            )
            source_path = args.reference_fingerprints
            source_key = reference_models[name]
            source_kind = "reference_array"
        else:
            source_path = prediction_paths[name]
            predictions = load_prediction_bundle(
                source_path,
                reference,
                metadata_path=prediction_metadata_paths.get(name),
                metadata_id_column=args.id_column,
            )
            source_key = "probs"
            source_kind = "standard_prediction_bundle"

        metrics = compute_per_spectrum_metrics(
            predictions.probabilities,
            reference.targets,
            thresholds[name],
            chunk_size=args.chunk_size,
        )
        metrics.insert(0, "threshold", thresholds[name])
        metrics.insert(0, "spectrum_id", reference.spectrum_ids)
        metrics.insert(0, "model", name)
        if predictions.inference_seconds is not None:
            metrics["inference_seconds"] = predictions.inference_seconds
        per_model[name] = metrics

        aggregates[name] = {
            "model": name,
            "threshold": thresholds[name],
            **aggregate_metrics(metrics),
        }
        molecule_frame = pd.DataFrame(
            {
                "molecule_id": molecule_ids,
                "fingerprint_tanimoto": metrics["fingerprint_tanimoto"],
            }
        )
        aggregates[name]["molecule_balanced_mean_fingerprint_tanimoto"] = float(
            molecule_frame.groupby("molecule_id")["fingerprint_tanimoto"].mean().mean()
        )
        if name in training_paths:
            training_ids = load_training_identifiers(training_paths[name])
            overlap_rows, overlap_rate = training_overlap(reference.metadata, training_ids)
            aggregates[name].update(
                {
                    "training_overlap_status": "checked",
                    "training_overlap_rows": overlap_rows,
                    "training_overlap_rate": overlap_rate,
                    "passes_training_overlap_check": overlap_rows == 0,
                }
            )
        else:
            aggregates[name].update(
                {
                    "training_overlap_status": "not_provided",
                    "training_overlap_rows": None,
                    "training_overlap_rate": None,
                    "passes_training_overlap_check": None,
                }
            )
        provenance[name] = {
            "kind": source_kind,
            "path": str(Path(source_path).resolve()),
            "array_key": source_key,
            "sha256": sha256_file(source_path),
            "metadata": (
                {
                    "path": str(Path(prediction_metadata_paths[name]).resolve()),
                    "sha256": sha256_file(prediction_metadata_paths[name]),
                }
                if name in prediction_metadata_paths
                else None
            ),
            "training_identifiers": (
                {
                    "path": str(Path(training_paths[name]).resolve()),
                    "sha256": sha256_file(training_paths[name]),
                }
                if name in training_paths
                else None
            ),
        }
        del predictions

    baseline_tanimoto = per_model[args.baseline]["fingerprint_tanimoto"].to_numpy()
    paired_frames = []
    for index, name in enumerate(model_names):
        if name == args.baseline:
            aggregates[name].update(
                {
                    "paired_mean_tanimoto_delta": 0.0,
                    "paired_median_tanimoto_delta": 0.0,
                    "paired_delta_ci95_low": 0.0,
                    "paired_delta_ci95_high": 0.0,
                    "paired_wins": 0,
                    "paired_losses": 0,
                    "paired_ties": len(reference.metadata),
                    "passes_minimum_gain": None,
                    "promotion_status": "baseline",
                }
            )
            continue
        deltas = per_model[name]["fingerprint_tanimoto"].to_numpy() - baseline_tanimoto
        ties = np.isclose(deltas, 0.0, rtol=0.0, atol=1e-12)
        ci_low, ci_high = paired_bootstrap_mean_ci(
            deltas,
            samples=args.bootstrap_samples,
            seed=args.seed + index,
            cluster_ids=molecule_ids,
        )
        mean_delta = float(deltas.mean())
        passes_gain = mean_delta >= args.minimum_gain and ci_low > 0.0
        overlap_status = aggregates[name]["passes_training_overlap_check"]
        if not passes_gain:
            promotion_status = "failed_quality_gate"
        elif overlap_status is None:
            promotion_status = "needs_training_overlap_evidence"
        elif not overlap_status:
            promotion_status = "failed_training_overlap_check"
        else:
            promotion_status = "passed_encoder_gate"
        aggregates[name].update(
            {
                "paired_mean_tanimoto_delta": mean_delta,
                "paired_median_tanimoto_delta": float(np.median(deltas)),
                "paired_delta_ci95_low": ci_low,
                "paired_delta_ci95_high": ci_high,
                "paired_wins": int(np.logical_and(deltas > 0.0, ~ties).sum()),
                "paired_losses": int(np.logical_and(deltas < 0.0, ~ties).sum()),
                "paired_ties": int(ties.sum()),
                "passes_minimum_gain": passes_gain,
                "promotion_status": promotion_status,
            }
        )
        paired_frames.append(
            pd.DataFrame(
                {
                    "model": name,
                    "baseline": args.baseline,
                    "spectrum_id": reference.spectrum_ids,
                    "molecule_id": molecule_ids,
                    "baseline_fingerprint_tanimoto": baseline_tanimoto,
                    "model_fingerprint_tanimoto": per_model[name][
                        "fingerprint_tanimoto"
                    ].to_numpy(),
                    "paired_tanimoto_delta": deltas,
                    "outcome": np.where(ties, "tie", np.where(deltas > 0.0, "win", "loss")),
                }
            )
        )

    aggregate_rows = sorted(
        aggregates.values(),
        key=lambda row: row["mean_fingerprint_tanimoto"],
        reverse=True,
    )
    for rank, row in enumerate(aggregate_rows, start=1):
        row["rank"] = rank

    metadata = reference.metadata.drop(columns=["fingerprint_index"], errors="ignore")
    per_spectrum_frames = []
    for name in model_names:
        frame = per_model[name].merge(
            metadata,
            left_on="spectrum_id",
            right_on=args.id_column,
            how="left",
            validate="one_to_one",
        )
        per_spectrum_frames.append(frame)
    per_spectrum = pd.concat(per_spectrum_frames, ignore_index=True)
    per_spectrum.to_csv(output_dir / "per_spectrum_metrics.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(output_dir / "aggregate_metrics.csv", index=False)
    if paired_frames:
        pd.concat(paired_frames, ignore_index=True).to_csv(
            output_dir / "paired_deltas.csv", index=False
        )

    stratified_rows = []
    for column in args.stratify_column:
        for (model, value), group in per_spectrum.groupby(["model", column], dropna=False):
            stratified_rows.append(
                {
                    "model": model,
                    "dimension": column,
                    "value": str(value),
                    "rows": int(len(group)),
                    "mean_fingerprint_tanimoto": float(group["fingerprint_tanimoto"].mean()),
                    "median_fingerprint_tanimoto": float(
                        group["fingerprint_tanimoto"].median()
                    ),
                    "mean_bce": float(group["bce"].mean()),
                }
            )
    if stratified_rows:
        pd.DataFrame(stratified_rows).to_csv(
            output_dir / "stratified_metrics.csv", index=False
        )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_code_revision": args.code_revision or git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "reference": {
            "metadata_path": str(Path(args.reference_metadata).resolve()),
            "metadata_sha256": sha256_file(args.reference_metadata),
            "fingerprints_path": str(Path(args.reference_fingerprints).resolve()),
            "fingerprints_sha256": sha256_file(args.reference_fingerprints),
            "target_key": args.target_key,
            "id_column": args.id_column,
            "ordered_spectrum_ids_sha256": hashlib.sha256(
                "\n".join(reference.spectrum_ids).encode("utf-8")
            ).hexdigest(),
            "rows": len(reference.metadata),
            "fingerprint_bits": reference.fingerprint_bits,
            "fingerprint_type": "Morgan",
            "fingerprint_radius": 2,
            "fingerprint_use_chirality": False,
        },
        "baseline": args.baseline,
        "minimum_gain": args.minimum_gain,
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "bootstrap_unit": "inchi_key_first_block_cluster",
        "stratify_columns": args.stratify_column,
        "models": provenance,
        "ranking": aggregate_rows,
        "outputs": {
            "aggregate_metrics": "aggregate_metrics.csv",
            "per_spectrum_metrics": "per_spectrum_metrics.csv",
            "paired_deltas": "paired_deltas.csv" if paired_frames else None,
            "stratified_metrics": "stratified_metrics.csv" if stratified_rows else None,
        },
    }
    (output_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps({"ranking": aggregate_rows}, indent=2))
    print(f"Saved benchmark outputs to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
