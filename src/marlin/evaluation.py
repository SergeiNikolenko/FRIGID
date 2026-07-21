"""Lightweight input loading for MARLIN evaluation."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


MIST_LANE_KIND = (
    "official MIST fingerprint probabilities from MIST-CF predicted formulas"
)
MIST_FORMULA_SOURCE = (
    "Highest-scoring SIRIUS-consistent MIST-CF candidate within 10 ppm; "
    "no ground-truth formula"
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
    sirius_bridge_manifest_path: Path,
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
        "sirius_bridge_manifest_sha256": sha256_file(sirius_bridge_manifest_path),
        "mist_labels_sha256": sha256_file(mist_labels_path),
    }
    mismatches = {
        key: {"expected": expected, "observed": payload.get(key)}
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "MIST lane provenance does not prove the formula-blind official lane: "
            f"{mismatches}"
        )
    for key in (
        "official_mist_git_commit",
        "sirius_version",
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
        "300_to_500": [
            row for row in rows if 300.0 <= row["neutral_mass"] < 500.0
        ],
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
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as arrays:
        if key not in arrays:
            raise KeyError(f"{key!r} not found in {path}; keys={arrays.files}")
        values = np.asarray(arrays[key])
        if "spectrum_ids" in arrays:
            positions = {str(value): index for index, value in enumerate(arrays["spectrum_ids"])}
            try:
                values = values[[positions[str(value)] for value in metadata["spec_name"]]]
            except KeyError as error:
                raise ValueError(f"fingerprint bundle is missing spectrum {error.args[0]}") from error
        elif allow_leading_subset and values.ndim == 2 and values.shape[0] >= len(metadata):
            values = values[: len(metadata)]
    if values.shape != (len(metadata), 4096):
        raise ValueError(
            f"fingerprints must have shape ({len(metadata)}, 4096), got {values.shape}"
        )
    if threshold is not None:
        values = values >= threshold
    elif not np.array_equal(values, values.astype(bool)):
        raise ValueError("non-binary fingerprints require --threshold")
    return values.astype(np.float32, copy=False)
