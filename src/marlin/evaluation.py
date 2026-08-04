"""Lightweight input loading for MARLIN evaluation."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


MIST_LANE_KIND = (
    "MIST fingerprint probabilities with MIST-CF top-1 subformula adapter"
)
MIST_FORMULA_SOURCE = (
    "Top-1 formula from connectivity-clean MIST-CF used only for "
    "peak-to-subformula features; no ground-truth formula"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_mist_lane_provenance(
    path: Path,
    expected_rows: int,
    fingerprint_path: Path,
    metadata_path: Path,
    formula_manifest_path: Path,
    feature_bridge_manifest_path: Path,
    mist_labels_path: Path,
) -> dict:
    payload = json.loads(path.read_text())
    required = {
        "kind": MIST_LANE_KIND,
        "formula_source": MIST_FORMULA_SOURCE,
        "rows": expected_rows,
        "fingerprint_bits": 4096,
        "output_sha256": sha256_file(fingerprint_path),
        "reference_metadata_sha256": sha256_file(metadata_path),
        "formula_manifest_sha256": sha256_file(formula_manifest_path),
        "feature_bridge_manifest_sha256": sha256_file(feature_bridge_manifest_path),
        "mist_labels_sha256": sha256_file(mist_labels_path),
    }
    mismatches = {
        key: {"expected": expected, "observed": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "MIST lane provenance does not prove the formula-blind adapted lane: "
            f"{mismatches}"
        )
    for key in (
        "official_mist_git_commit",
        "mist_cf_git_commit",
        "mist_cf_checkpoint_sha256",
        "mist_checkpoint_sha256",
    ):
        if not payload.get(key):
            raise ValueError(f"MIST lane provenance is missing {key}")
    return payload


def mean_metric(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows])) if rows else float("nan")


def mass_bin_metrics(rows: list[dict]) -> dict[str, dict[str, float | int]]:
    bins = {
        "lt_300": [row for row in rows if row["neutral_mass"] < 300.0],
        "300_to_500": [row for row in rows if 300.0 <= row["neutral_mass"] < 500.0],
        "gte_500": [row for row in rows if row["neutral_mass"] >= 500.0],
    }
    return {
        name: {
            "rows": len(bin_rows),
            "exact_top1": mean_metric(bin_rows, "exact_top1"),
        }
        for name, bin_rows in bins.items()
    }


def load_fingerprints(
    path: Path,
    key: str,
    threshold: float | None,
    metadata: pd.DataFrame,
    *,
    allow_leading_subset: bool = False,
    preserve_probabilities: bool = False,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as arrays:
        if key not in arrays:
            raise KeyError(f"{key!r} not found in {path}; keys={arrays.files}")
        values = np.asarray(arrays[key])
        if "spectrum_ids" in arrays:
            positions = {
                str(value): index for index, value in enumerate(arrays["spectrum_ids"])
            }
            try:
                values = values[
                    [positions[str(value)] for value in metadata["spec_name"]]
                ]
            except KeyError as error:
                raise ValueError(
                    f"fingerprint bundle is missing spectrum {error.args[0]}"
                ) from error
        elif (
            allow_leading_subset
            and values.ndim == 2
            and values.shape[0] >= len(metadata)
        ):
            values = values[: len(metadata)]
    if values.shape != (len(metadata), 4096):
        raise ValueError(
            f"fingerprints must have shape ({len(metadata)}, 4096), got {values.shape}"
        )
    if not preserve_probabilities:
        if threshold is not None:
            values = values >= threshold
        elif not np.array_equal(values, values.astype(bool)):
            raise ValueError("non-binary fingerprints require --threshold")
    else:
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError("soft fingerprints must be probabilities in [0, 1]")
        if threshold is not None:
            # The conditioning encoder activates every bit above 0.5, so a soft
            # bundle needs the threshold applied as a sparsity gate; keeping the
            # amplitude only on surviving bits is what --soft-fingerprint means.
            values = np.where(values >= threshold, values, 0.0)
    return values.astype(np.float32, copy=False)
