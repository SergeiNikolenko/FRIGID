"""Shared frozen-encoder probe for Morgan fingerprint prediction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmbeddingBundle:
    """Dense spectrum embeddings and their row-aligned benchmark fields."""

    embeddings: np.ndarray
    targets: np.ndarray
    spectrum_ids: np.ndarray
    structure_ids: np.ndarray
    inference_seconds: np.ndarray | None


def _decode_strings(values: np.ndarray, *, source: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"{source} must be one-dimensional, got {values.shape}")
    decoded = []
    for value in values.tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        text = str(value).strip()
        if not text or text.lower() == "nan":
            raise ValueError(f"{source} contains an empty value")
        decoded.append(text)
    return np.asarray(decoded, dtype=str)


def normalize_structure_ids(inchikeys: np.ndarray) -> np.ndarray:
    """Normalize InChIKeys to the connectivity block used for leakage checks."""

    decoded = _decode_strings(inchikeys, source="inchikeys")
    return np.asarray(
        [value.split("-", maxsplit=1)[0].upper() for value in decoded], dtype=str
    )


def load_embedding_bundle(
    path: str,
    *,
    fingerprint_bits: int = 4096,
) -> EmbeddingBundle:
    """Load and strictly validate an embedding/target NPZ bundle."""

    with np.load(path, allow_pickle=False) as arrays:
        required = {"embeddings", "ground_truth", "spectrum_ids", "inchikeys"}
        missing = sorted(required - set(arrays.files))
        if missing:
            raise ValueError(
                f"Embedding bundle {path} is missing keys {missing}; "
                f"available keys: {arrays.files}"
            )
        embeddings = np.asarray(arrays["embeddings"], dtype=np.float32)
        targets = np.asarray(arrays["ground_truth"])
        spectrum_ids = _decode_strings(
            arrays["spectrum_ids"], source=f"{path}:spectrum_ids"
        )
        structure_ids = normalize_structure_ids(arrays["inchikeys"])
        inference_seconds = None
        if "inference_seconds" in arrays:
            raw_seconds = np.asarray(arrays["inference_seconds"], dtype=np.float64)
            if raw_seconds.ndim == 0:
                raw_seconds = np.full(len(spectrum_ids), float(raw_seconds) / len(spectrum_ids))
            inference_seconds = raw_seconds

    expected_rows = len(spectrum_ids)
    if expected_rows == 0:
        raise ValueError(f"Embedding bundle is empty: {path}")
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows:
        raise ValueError(
            f"{path}:embeddings must have shape [N, d] with N={expected_rows}, "
            f"got {embeddings.shape}"
        )
    if embeddings.shape[1] <= 0 or not np.isfinite(embeddings).all():
        raise ValueError(f"{path}:embeddings must be finite with a positive width")
    expected_target_shape = (expected_rows, fingerprint_bits)
    if targets.shape != expected_target_shape:
        raise ValueError(
            f"{path}:ground_truth has shape {targets.shape}; "
            f"expected {expected_target_shape}"
        )
    binary = np.logical_or(np.isclose(targets, 0.0), np.isclose(targets, 1.0))
    if not binary.all():
        raise ValueError(f"{path}:ground_truth must contain only binary values")
    if len(set(spectrum_ids.tolist())) != expected_rows:
        raise ValueError(f"{path}:spectrum_ids contains duplicates")
    if len(structure_ids) != expected_rows:
        raise ValueError(f"{path}:inchikeys row count does not match spectrum_ids")
    if inference_seconds is not None:
        if inference_seconds.shape != (expected_rows,):
            raise ValueError(
                f"{path}:inference_seconds has shape {inference_seconds.shape}; "
                f"expected {(expected_rows,)}"
            )
        if not np.isfinite(inference_seconds).all() or np.any(inference_seconds < 0.0):
            raise ValueError(f"{path}:inference_seconds must be finite and non-negative")

    return EmbeddingBundle(
        embeddings=embeddings,
        targets=(targets > 0.5).astype(np.uint8, copy=False),
        spectrum_ids=spectrum_ids,
        structure_ids=structure_ids,
        inference_seconds=inference_seconds,
    )


def deterministic_group_holdout(
    structure_ids: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indexes by whole structure clusters with a stable hash."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            f"validation_fraction must be in (0, 1), got {validation_fraction}"
        )
    structure_ids = _decode_strings(structure_ids, source="structure_ids")
    unique_structures = sorted(set(structure_ids.tolist()))
    if len(unique_structures) < 2:
        raise ValueError("At least two structure clusters are required")
    validation_count = int(round(len(unique_structures) * validation_fraction))
    validation_count = min(max(validation_count, 1), len(unique_structures) - 1)
    ranked = sorted(
        unique_structures,
        key=lambda value: hashlib.sha256(
            f"{seed}\0{value}".encode("utf-8")
        ).digest(),
    )
    validation_structures = set(ranked[:validation_count])
    validation_mask = np.asarray(
        [value in validation_structures for value in structure_ids], dtype=bool
    )
    return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)


def global_positive_weight(targets: np.ndarray, row_indexes: np.ndarray) -> float:
    """Return the global negative/positive ratio without copying the full matrix."""

    targets = np.asarray(targets)
    row_indexes = np.asarray(row_indexes, dtype=np.int64)
    if targets.ndim != 2 or row_indexes.ndim != 1:
        raise ValueError("targets must be 2D and row_indexes must be 1D")
    if len(row_indexes) == 0:
        raise ValueError("Cannot calculate a positive weight for an empty split")
    positives = 0
    for start in range(0, len(row_indexes), 8192):
        positives += int(targets[row_indexes[start : start + 8192]].sum())
    total = int(len(row_indexes) * targets.shape[1])
    if positives == 0:
        raise ValueError("Training targets contain no positive fingerprint bits")
    return float((total - positives) / positives)
