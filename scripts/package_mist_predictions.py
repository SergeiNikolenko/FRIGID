#!/usr/bin/env python3
"""Package official MIST Morgan-4096 probabilities in benchmark order."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


FORMULA_SOURCE = (
    "Highest-scoring SIRIUS-consistent MIST-CF candidate within 10 ppm; "
    "no ground-truth formula"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_predictions(
    predictions: Path,
    metadata: Path,
    formula_manifest: Path,
    sirius_bridge_manifest: Path,
    mist_labels: Path,
    output: Path,
    official_mist_git_commit: str,
    sirius_version: str,
    mist_checkpoint_sha256: str,
) -> dict:
    with predictions.open("rb") as handle:
        payload = pickle.load(handle)
    expected_dataset_name = mist_labels.parent.name
    if payload.get("dataset_name") != expected_dataset_name:
        raise ValueError(
            f"MIST pickle dataset mismatch: expected={expected_dataset_name} "
            f"observed={payload.get('dataset_name')}"
        )
    if payload.get("args", {}).get("labels_name") != mist_labels.name:
        raise ValueError(
            f"MIST pickle labels mismatch: expected={mist_labels.name} "
            f"observed={payload.get('args', {}).get('labels_name')}"
        )
    names = [str(value) for value in payload["names"]]
    probabilities = np.asarray(payload["preds"], dtype=np.float32)
    if probabilities.shape != (len(names), 4096):
        raise ValueError(f"Expected Morgan-4096 probabilities, got {probabilities.shape}")
    if len(names) != len(set(names)):
        raise ValueError("MIST predictions contain duplicate spectrum IDs")
    ordered = pd.read_csv(metadata).sort_values("fingerprint_index", kind="stable")
    expected = ordered["spec_name"].astype(str).tolist()
    positions = {name: index for index, name in enumerate(names)}
    missing = [name for name in expected if name not in positions]
    extra = sorted(set(names) - set(expected))
    if missing or extra:
        raise ValueError(
            f"MIST/metadata ID mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    values = probabilities[[positions[name] for name in expected]]
    formula_evidence = json.loads(formula_manifest.read_text())
    sirius_evidence = json.loads(sirius_bridge_manifest.read_text())
    if (
        formula_evidence.get("kind")
        != "Mass-consistent MIST-CF predicted-formula bridge into official MIST"
        or formula_evidence.get("formula_source") != FORMULA_SOURCE
        or formula_evidence.get("precursor_ppm_tolerance") != 10.0
        or formula_evidence.get("rows") != len(expected)
        or not formula_evidence.get("sirius_consistency_validated")
    ):
        raise ValueError("Formula manifest does not prove the formula-blind MIST-CF lane")
    maximum_ppm_error = sirius_evidence.get("maximum_mist_precursor_ppm_error")
    required_sirius_fields = (
        "sirius_audit_sha256",
        "summary_sha256",
        "sirius_tree_evidence_sha256",
        "per_id_mapping_sha256",
    )
    if (
        sirius_evidence.get("kind")
        != "SIRIUS-validated formula-blind bridge into official MIST"
        or sirius_evidence.get("formula_source") != FORMULA_SOURCE
        or sirius_evidence.get("rows") != len(expected)
        or sirius_evidence.get("formula_manifest_sha256")
        != sha256_file(formula_manifest)
        or sirius_evidence.get("mist_labels_sha256") != sha256_file(mist_labels)
        or not isinstance(maximum_ppm_error, (int, float))
        or not math.isfinite(maximum_ppm_error)
        or maximum_ppm_error > 10.0
        or any(not sirius_evidence.get(key) for key in required_sirius_fields)
    ):
        raise ValueError("Post-SIRIUS manifest does not prove a mass-consistent MIST bridge")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, probs=values, spectrum_ids=np.asarray(expected))
    manifest = {
        "schema_version": 1,
        "kind": "official MIST fingerprint probabilities from MIST-CF predicted formulas",
        "rows": len(expected),
        "fingerprint_bits": 4096,
        "formula_source": formula_evidence["formula_source"],
        "precursor_ppm_tolerance": formula_evidence["precursor_ppm_tolerance"],
        "fallback_rows": formula_evidence["fallback_rows"],
        "maximum_candidate_rank": formula_evidence["maximum_candidate_rank"],
        "predictions_pickle_sha256": sha256_file(predictions),
        "reference_metadata_sha256": sha256_file(metadata),
        "formula_manifest_sha256": sha256_file(formula_manifest),
        "sirius_bridge_manifest_sha256": sha256_file(sirius_bridge_manifest),
        "mist_labels_sha256": sha256_file(mist_labels),
        "official_mist_git_commit": official_mist_git_commit,
        "sirius_version": sirius_version,
        "mist_checkpoint_sha256": mist_checkpoint_sha256,
        "output_sha256": sha256_file(output),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--formula-manifest", type=Path, required=True)
    parser.add_argument("--sirius-bridge-manifest", type=Path, required=True)
    parser.add_argument("--mist-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-mist-git-commit", required=True)
    parser.add_argument("--sirius-version", required=True)
    parser.add_argument("--mist-checkpoint-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            package_predictions(
                args.predictions,
                args.metadata,
                args.formula_manifest,
                args.sirius_bridge_manifest,
                args.mist_labels,
                args.output,
                args.official_mist_git_commit,
                args.sirius_version,
                args.mist_checkpoint_sha256,
            )
        )
    )


if __name__ == "__main__":
    main()
