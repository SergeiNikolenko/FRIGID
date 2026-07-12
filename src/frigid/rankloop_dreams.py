"""Deterministic DreaMS input and output helpers for RankLoop."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def merge_spectrum_peaks(
    spectra: Sequence[np.ndarray],
) -> np.ndarray:
    """Merge all MS2 arrays into one deterministic m/z-sorted peak list."""
    arrays: list[np.ndarray] = []
    for peaks in spectra:
        peak_array = np.asarray(peaks, dtype=np.float64)
        if peak_array.size == 0:
            continue
        if peak_array.ndim != 2 or peak_array.shape[1] != 2:
            raise ValueError(f"Expected an N x 2 peak array, got {peak_array.shape}.")
        arrays.append(peak_array)
    if not arrays:
        raise ValueError("Spectrum has no peaks.")

    merged = np.concatenate(arrays, axis=0)
    if not np.isfinite(merged).all():
        raise ValueError("Spectrum contains non-finite peak values.")
    return merged[np.argsort(merged[:, 0], kind="stable")]


def build_dreams_mgf_entry(
    *,
    spec_name: str,
    formula: str,
    precursor_mz: float,
    peaks: np.ndarray,
) -> str:
    """Serialize one spectrum in the MGF form consumed by the DreaMS API."""
    if "\n" in spec_name or "\r" in spec_name:
        raise ValueError("spec_name cannot contain line breaks.")
    if not np.isfinite(precursor_mz) or precursor_mz <= 0:
        raise ValueError(f"Invalid precursor m/z for {spec_name}: {precursor_mz}")
    peak_array = np.asarray(peaks, dtype=np.float64)
    if peak_array.ndim != 2 or peak_array.shape[1] != 2 or len(peak_array) == 0:
        raise ValueError(f"Expected a non-empty N x 2 peak array, got {peak_array.shape}.")
    if not np.isfinite(peak_array).all():
        raise ValueError("Spectrum contains non-finite peak values.")

    rows = [
        "BEGIN IONS",
        f"TITLE={spec_name}",
        f"NAME={spec_name}",
        f"SCANS={spec_name}",
        f"PEPMASS={precursor_mz:.10g}",
        f"FORMULA={formula}",
    ]
    rows.extend(f"{mz:.10g} {intensity:.10g}" for mz, intensity in peak_array)
    rows.append("END IONS")
    return "\n".join(rows)


def validate_dreams_embeddings(
    embeddings: np.ndarray,
    *,
    expected_rows: int,
    expected_dimension: int | None = None,
) -> np.ndarray:
    """Validate and normalize a DreaMS embedding matrix for RankLoop."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D DreaMS embedding matrix, got {matrix.shape}.")
    if matrix.shape[0] != expected_rows:
        raise ValueError(
            "DreaMS embedding row count does not match metadata: "
            f"{matrix.shape[0]} != {expected_rows}."
        )
    if expected_dimension is not None and matrix.shape[1] != expected_dimension:
        raise ValueError(
            "Unexpected DreaMS embedding dimension: "
            f"{matrix.shape[1]} != {expected_dimension}."
        )
    if not np.isfinite(matrix).all():
        raise ValueError("DreaMS embeddings contain non-finite values.")
    return matrix
