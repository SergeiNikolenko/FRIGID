"""Lightweight input loading for MARLIN evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


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
