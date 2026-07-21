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
    "Top-1 formula from connectivity-clean MIST-CF used only for "
    "peak-to-subformula features; no ground-truth formula"
)
FORMULA_MANIFEST_KIND = "MIST-CF top-1 predicted-formula inputs for official MIST"
FEATURE_BRIDGE_KIND = "MIST-CF top-1 peak-to-subformula bridge into official MIST"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def package_predictions(
    predictions: Path,
    metadata: Path,
    formula_manifest: Path,
    feature_bridge_manifest: Path,
    mist_labels: Path,
    output: Path,
    official_mist_git_commit: str,
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
        raise ValueError(
            f"Expected Morgan-4096 probabilities, got {probabilities.shape}"
        )
    if len(names) != len(set(names)):
        raise ValueError("MIST predictions contain duplicate spectrum IDs")
    ordered = pd.read_csv(metadata).sort_values("fingerprint_index", kind="stable")
    expected = ordered["spec_name"].astype(str).tolist()
    label_ids = pd.read_csv(mist_labels, sep="\t")["spec"].astype(str).tolist()
    if label_ids != expected:
        raise ValueError("MIST labels are not in the locked metadata order")
    positions = {name: index for index, name in enumerate(names)}
    missing = [name for name in expected if name not in positions]
    extra = sorted(set(names) - set(expected))
    if missing or extra:
        raise ValueError(
            f"MIST/metadata ID mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    values = probabilities[[positions[name] for name in expected]]
    formula_evidence = json.loads(formula_manifest.read_text())
    feature_evidence = json.loads(feature_bridge_manifest.read_text())
    bridge_root = feature_bridge_manifest.parent
    evidence_path = bridge_root / "feature_bridge_rows.jsonl"
    summary_path = bridge_root / "sirius_outputs/summary_statistics/summary_df.tsv"
    if (
        formula_evidence.get("kind") != FORMULA_MANIFEST_KIND
        or formula_evidence.get("formula_source") != FORMULA_SOURCE
        or formula_evidence.get("precursor_ppm_tolerance") != 10.0
        or formula_evidence.get("rows") != len(expected)
        or formula_evidence.get("fallback_rows") != 0
        or formula_evidence.get("maximum_candidate_rank") != 1
    ):
        raise ValueError(
            "Formula manifest does not prove the formula-blind MIST-CF lane"
        )
    maximum_ppm_error = feature_evidence.get("maximum_precursor_ppm_error")
    maximum_assignment_ppm = feature_evidence.get("maximum_subformula_assignment_ppm")
    root_only_ids = feature_evidence.get("root_only_ids")
    subformula_dir_value = feature_evidence.get("mist_cf_subformula_dir")
    required_feature_fields = (
        "summary_sha256",
        "mist_cf_subformula_evidence_sha256",
        "peakformula_tree_evidence_sha256",
        "per_id_mapping_sha256",
        "mist_cf_git_commit",
        "mist_cf_checkpoint_sha256",
    )
    if (
        feature_evidence.get("kind") != FEATURE_BRIDGE_KIND
        or feature_evidence.get("formula_source") != FORMULA_SOURCE
        or feature_evidence.get("rows") != len(expected)
        or feature_evidence.get("top1_candidate_rows") != len(expected)
        or feature_evidence.get("formula_manifest_sha256")
        != sha256_file(formula_manifest)
        or feature_evidence.get("mist_labels_sha256") != sha256_file(mist_labels)
        or not isinstance(maximum_ppm_error, (int, float))
        or not math.isfinite(maximum_ppm_error)
        or maximum_ppm_error > 10.0
        or not isinstance(maximum_assignment_ppm, (int, float))
        or not math.isfinite(maximum_assignment_ppm)
        or maximum_assignment_ppm > 15.0
        or feature_evidence.get("subformula_assignment_ppm_tolerance") != 15.0
        or not isinstance(feature_evidence.get("root_only_rows"), int)
        or feature_evidence.get("root_only_rows", -1) < 0
        or not isinstance(root_only_ids, list)
        or len(root_only_ids) != feature_evidence.get("root_only_rows")
        or len(root_only_ids) != len(set(root_only_ids))
        or not set(root_only_ids).issubset(expected)
        or not isinstance(subformula_dir_value, str)
        or any(not feature_evidence.get(key) for key in required_feature_fields)
    ):
        raise ValueError("Feature manifest does not prove the top-1 MIST-CF bridge")
    if (
        not evidence_path.is_file()
        or sha256_file(evidence_path) != feature_evidence["per_id_mapping_sha256"]
        or not summary_path.is_file()
        or sha256_file(summary_path) != feature_evidence["summary_sha256"]
    ):
        raise ValueError("Feature bridge files do not match their manifest")
    with evidence_path.open() as handle:
        evidence_rows = [json.loads(line) for line in handle if line.strip()]
    if [row.get("spec") for row in evidence_rows] != expected:
        raise ValueError("Feature bridge rows are not in the locked metadata order")
    subformula_dir = Path(subformula_dir_value)
    for row in evidence_rows:
        spectrum_id = row["spec"]
        subformula_path = subformula_dir / f"{spectrum_id}.json"
        if sha256_file(subformula_path) != row.get("subformula_file_sha256"):
            raise ValueError(f"MIST-CF subformula evidence changed for {spectrum_id}")
        selected = json.loads(subformula_path.read_text()).get(row["formula"])
        if selected is None or json_digest(selected) != row.get(
            "selected_subformula_sha256"
        ):
            raise ValueError(f"Selected MIST-CF subformula changed for {spectrum_id}")
        if sha256_file(
            bridge_root / "peakformula_trees" / f"{spectrum_id}.json"
        ) != row.get("peakformula_tree_sha256"):
            raise ValueError(f"PeakFormula tree changed for {spectrum_id}")
    if feature_evidence["mist_cf_subformula_evidence_sha256"] != json_digest(
        [
            {
                "spec": row["spec"],
                "subformula_file_sha256": row["subformula_file_sha256"],
                "selected_subformula_sha256": row["selected_subformula_sha256"],
            }
            for row in evidence_rows
        ]
    ) or feature_evidence["peakformula_tree_evidence_sha256"] != json_digest(
        [
            {
                "spec": row["spec"],
                "peakformula_tree_sha256": row["peakformula_tree_sha256"],
            }
            for row in evidence_rows
        ]
    ):
        raise ValueError("Feature bridge aggregate digests do not match its rows")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, probs=values, spectrum_ids=np.asarray(expected))
    manifest = {
        "schema_version": 1,
        "kind": "MIST fingerprint probabilities with MIST-CF top-1 subformula adapter",
        "rows": len(expected),
        "fingerprint_bits": 4096,
        "formula_source": formula_evidence["formula_source"],
        "precursor_ppm_tolerance": formula_evidence["precursor_ppm_tolerance"],
        "fallback_rows": formula_evidence["fallback_rows"],
        "maximum_candidate_rank": formula_evidence["maximum_candidate_rank"],
        "predictions_pickle_sha256": sha256_file(predictions),
        "reference_metadata_sha256": sha256_file(metadata),
        "formula_manifest_sha256": sha256_file(formula_manifest),
        "feature_bridge_manifest_sha256": sha256_file(feature_bridge_manifest),
        "mist_labels_sha256": sha256_file(mist_labels),
        "official_mist_git_commit": official_mist_git_commit,
        "mist_cf_git_commit": feature_evidence["mist_cf_git_commit"],
        "mist_cf_checkpoint_sha256": feature_evidence["mist_cf_checkpoint_sha256"],
        "root_only_rows": feature_evidence["root_only_rows"],
        "root_only_ids": root_only_ids,
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
    parser.add_argument("--feature-bridge-manifest", type=Path, required=True)
    parser.add_argument("--mist-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-mist-git-commit", required=True)
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
                args.feature_bridge_manifest,
                args.mist_labels,
                args.output,
                args.official_mist_git_commit,
                args.mist_checkpoint_sha256,
            )
        )
    )


if __name__ == "__main__":
    main()
