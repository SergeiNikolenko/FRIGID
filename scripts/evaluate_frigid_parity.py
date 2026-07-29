#!/usr/bin/env python3
"""Evaluate the released FRIGID sampler on MARLIN fingerprint inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# The official sampler imports SAFE before RDKit drawing libraries.
from dlm.sampler import Sampler
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors

from evaluate_marlin_nplib1 import (
    add_formula_metrics,
    connectivity,
    formula_metric_summary,
    morgan,
    publish_clearml_evaluation,
)
from marlin.evaluation import load_fingerprints, mass_bin_metrics, mean_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--fingerprints", type=Path, required=True)
    parser.add_argument("--fingerprint-key", required=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--max-spectra", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--randomness", type=float, default=0.5)
    parser.add_argument("--ppm-tolerance", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clearml-task-id")
    parser.add_argument("--clearml-iteration", type=int, default=0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def explicit_fingerprint(bits: np.ndarray) -> DataStructs.ExplicitBitVect:
    fingerprint = DataStructs.ExplicitBitVect(int(bits.shape[0]))
    fingerprint.SetBitsFromList(np.flatnonzero(bits).astype(int).tolist())
    return fingerprint


def internal_diversity(candidates: list[dict]) -> float | None:
    fingerprints = []
    for candidate in candidates:
        molecule = Chem.MolFromSmiles(candidate["smiles"])
        if molecule is not None:
            fingerprints.append(morgan(molecule))
    distances = [
        1.0 - DataStructs.TanimotoSimilarity(fingerprints[left], fingerprints[right])
        for left in range(len(fingerprints))
        for right in range(left + 1, len(fingerprints))
    ]
    return float(np.mean(distances)) if distances else None


def main() -> None:
    args = parse_args()
    if args.candidates <= 0:
        raise ValueError("--candidates must be positive")
    if args.max_spectra <= 0:
        raise ValueError("--max-spectra must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if args.randomness < 0:
        raise ValueError("--randomness must be non-negative")
    if args.ppm_tolerance <= 0:
        raise ValueError("--ppm-tolerance must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = pd.read_csv(args.metadata).iloc[: args.max_spectra].copy()
    fingerprints = load_fingerprints(
        args.fingerprints,
        args.fingerprint_key,
        args.threshold,
        metadata,
        allow_leading_subset=True,
    )
    settings = {
        "sampler": "official FRIGID confidence sampler",
        "fingerprint_key": args.fingerprint_key,
        "fingerprint_threshold": args.threshold,
        "candidates": args.candidates,
        "max_spectra": args.max_spectra,
        "temperature": args.temperature,
        "randomness": args.randomness,
        "ppm_tolerance": args.ppm_tolerance,
        "seed": args.seed,
        "length_source": "official sampler data/len.pk or its 20..99 fallback",
    }
    signature = {
        "schema_version": 1,
        "git_commit": git_commit(),
        "settings": settings,
        "inputs": {
            str(path.resolve()): {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in (args.checkpoint, args.metadata, args.fingerprints)
        },
    }
    (args.output_dir / "run_signature.json").write_text(
        json.dumps(signature, indent=2, sort_keys=True) + "\n"
    )

    sampler = Sampler(str(args.checkpoint))
    rows: list[dict] = []
    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w") as output:
        for position, record in metadata.reset_index(drop=True).iterrows():
            sample_seed = args.seed + position
            random.seed(sample_seed)
            np.random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed)
                torch.cuda.synchronize()
            started = time.perf_counter()
            generated = sampler.fingerprint_conditioned_generation(
                fingerprints[position],
                num_samples=args.candidates,
                softmax_temp=args.temperature,
                randomness=args.randomness,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started

            target_molecule = Chem.MolFromSmiles(record["smiles"])
            if target_molecule is None:
                raise ValueError(f"invalid target SMILES: {record['smiles']}")
            target_fingerprint = morgan(target_molecule)
            predicted_fingerprint = explicit_fingerprint(fingerprints[position])
            target_connectivity = str(record["inchikey_first_block"])
            neutral_mass = float(record["neutral_mass"])
            valid = []
            for smiles in generated:
                molecule = Chem.MolFromSmiles(smiles)
                if molecule is None:
                    continue
                canonical = Chem.MolToSmiles(molecule, canonical=True)
                fingerprint = morgan(molecule)
                mass_error_ppm = (
                    abs(rdMolDescriptors.CalcExactMolWt(molecule) - neutral_mass)
                    / neutral_mass
                    * 1e6
                )
                valid.append(
                    {
                        "smiles": canonical,
                        "safe": None,
                        "predicted_fingerprint_tanimoto": (
                            DataStructs.TanimotoSimilarity(
                                predicted_fingerprint, fingerprint
                            )
                        ),
                        "target_fingerprint_tanimoto": (
                            DataStructs.TanimotoSimilarity(
                                target_fingerprint, fingerprint
                            )
                        ),
                        "mass_error_ppm": mass_error_ppm,
                        "exact_connectivity": (
                            connectivity(canonical) == target_connectivity
                        ),
                        "formula": rdMolDescriptors.CalcMolFormula(molecule),
                    }
                )
            valid_count = len(valid)
            unique = {
                candidate["smiles"]: candidate for candidate in valid
            }
            ranked = sorted(
                unique.values(),
                key=lambda candidate: (
                    -candidate["predicted_fingerprint_tanimoto"],
                    candidate["mass_error_ppm"],
                    candidate["smiles"],
                ),
            )
            mass_valid = sum(
                candidate["mass_error_ppm"] <= args.ppm_tolerance
                for candidate in valid
            )
            top_ten = ranked[:10]
            result = {
                "spec_name": str(record["spec_name"]),
                "lane": "dreams-ground-truth-fingerprint",
                "target_smiles": record["smiles"],
                "target_inchikey_first_block": target_connectivity,
                "neutral_mass": neutral_mass,
                "runtime_seconds": elapsed,
                "attempts": args.candidates,
                "valid": valid_count,
                "mass_valid": mass_valid,
                "unique_mass_valid": len(
                    {
                        candidate["smiles"]
                        for candidate in valid
                        if candidate["mass_error_ppm"] <= args.ppm_tolerance
                    }
                ),
                "constraint_dead_ends": 0,
                "eos_terminated": 0,
                "max_length_terminated": 0,
                "sample_terminal_safes": [],
                "sample_dead_ends": [],
                "validity": valid_count / args.candidates,
                "mass_validity": mass_valid / max(valid_count, 1),
                "uniqueness": len(unique) / max(valid_count, 1),
                "candidate_returned": bool(ranked),
                "exact_top1": bool(
                    ranked and ranked[0]["exact_connectivity"]
                ),
                "exact_top10": any(
                    candidate["exact_connectivity"] for candidate in top_ten
                ),
                "tanimoto_top1": (
                    ranked[0]["target_fingerprint_tanimoto"] if ranked else 0.0
                ),
                "tanimoto_top10": max(
                    (
                        candidate["target_fingerprint_tanimoto"]
                        for candidate in top_ten
                    ),
                    default=0.0,
                ),
                "candidates": ranked,
            }
            add_formula_metrics(
                result,
                target_formula=rdMolDescriptors.CalcMolFormula(target_molecule),
            )
            output.write(json.dumps(result, sort_keys=True) + "\n")
            output.flush()
            rows.append(result)
            print(
                f"{position + 1}/{len(metadata)} {result['spec_name']} "
                f"valid={valid_count}/{args.candidates} unique={len(unique)} "
                f"runtime={elapsed:.3f}s",
                flush=True,
            )

    rows_with_candidate = [row for row in rows if row["candidate_returned"]]
    diversities = [
        value
        for row in rows
        if (value := internal_diversity(row["candidates"])) is not None
    ]
    metrics = {
        "lane": "dreams-ground-truth-fingerprint",
        "rows": len(rows),
        "exact_top1": mean_metric(rows, "exact_top1"),
        "exact_top10": mean_metric(rows, "exact_top10"),
        "candidate_return_rate": len(rows_with_candidate) / max(len(rows), 1),
        "tanimoto_top1": mean_metric(rows_with_candidate, "tanimoto_top1"),
        "tanimoto_top10": mean_metric(rows_with_candidate, "tanimoto_top10"),
        **formula_metric_summary(rows),
        "mass_bins": mass_bin_metrics(rows),
        "validity": mean_metric(rows, "validity"),
        "mass_validity": mean_metric(rows, "mass_validity"),
        "uniqueness": mean_metric(rows, "uniqueness"),
        "internal_diversity": float(np.mean(diversities)) if diversities else 0.0,
        "constraint_dead_ends_mean": 0.0,
        "eos_terminated_mean": 0.0,
        "max_length_terminated_mean": 0.0,
        "runtime_seconds_total": float(
            sum(row["runtime_seconds"] for row in rows)
        ),
        "runtime_seconds_mean": mean_metric(rows, "runtime_seconds"),
        "settings": settings,
        "metric_denominators": {
            "validity": "decoded RDKit-valid molecules / requested samples",
            "mass_validity": "within ppm tolerance / RDKit-valid molecules",
            "uniqueness": "unique canonical SMILES / RDKit-valid molecules",
            "tanimoto_top1": "rows with a returned candidate",
            "tanimoto_top10": "rows with a returned candidate",
            "internal_diversity": "mean within-spectrum pairwise Morgan distance",
        },
    }
    clearml = publish_clearml_evaluation(
        project_name=None,
        task_name=None,
        tags=[],
        metrics=metrics,
        rows=rows,
        settings=settings,
        task_id=args.clearml_task_id,
        iteration=args.clearml_iteration,
        evaluation_label="FRIGID parity",
    )
    if clearml is not None:
        metrics["clearml"] = clearml
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
