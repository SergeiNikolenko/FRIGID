"""Leakage-aware metrics and I/O for spectrum encoder fingerprint benchmarks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


STANDARD_PROBABILITIES_KEY = "probs"
STANDARD_SPECTRUM_IDS_KEY = "spectrum_ids"
STANDARD_INFERENCE_SECONDS_KEY = "inference_seconds"


@dataclass(frozen=True)
class ReferenceBundle:
    """Row-aligned benchmark metadata and binary target fingerprints."""

    metadata: pd.DataFrame
    spectrum_ids: np.ndarray
    targets: np.ndarray
    fingerprint_bits: int


@dataclass(frozen=True)
class PredictionBundle:
    """Predictions aligned to a :class:`ReferenceBundle`."""

    probabilities: np.ndarray
    inference_seconds: np.ndarray | None


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decode_ids(values: np.ndarray, *, source: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"{source} IDs must be one-dimensional, got {values.shape}")

    decoded = []
    for value in values.tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        text = str(value).strip()
        if not text or text.lower() == "nan":
            raise ValueError(f"{source} contains an empty spectrum ID")
        decoded.append(text)

    result = np.asarray(decoded, dtype=str)
    if len(set(result.tolist())) != len(result):
        duplicates = pd.Series(result)[pd.Series(result).duplicated()].unique().tolist()
        raise ValueError(f"{source} contains duplicate spectrum IDs: {duplicates[:5]}")
    return result


def _validate_targets(targets: np.ndarray, expected_rows: int, source: str) -> np.ndarray:
    targets = np.asarray(targets)
    if targets.ndim != 2:
        raise ValueError(f"{source} targets must be two-dimensional, got {targets.shape}")
    if targets.shape[0] != expected_rows:
        raise ValueError(
            f"{source} target rows ({targets.shape[0]}) do not match metadata ({expected_rows})"
        )
    if not np.isfinite(targets).all():
        raise ValueError(f"{source} targets contain NaN or infinite values")
    binary = np.logical_or(np.isclose(targets, 0.0), np.isclose(targets, 1.0))
    if not binary.all():
        raise ValueError(f"{source} targets must contain only binary fingerprint values")
    return targets > 0.5


def _validate_probabilities(
    probabilities: np.ndarray,
    *,
    expected_rows: int,
    fingerprint_bits: int,
    source: str,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    expected_shape = (expected_rows, fingerprint_bits)
    if probabilities.shape != expected_shape:
        raise ValueError(
            f"{source} probabilities have shape {probabilities.shape}; expected {expected_shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{source} probabilities contain NaN or infinite values")
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError(f"{source} probabilities must be in the closed interval [0, 1]")
    return probabilities


def load_reference_bundle(
    metadata_path: str | Path,
    fingerprints_path: str | Path,
    *,
    target_key: str = "ground_truth",
    id_column: str = "spec_name",
    index_column: str = "fingerprint_index",
) -> ReferenceBundle:
    """Load a FRIGID fingerprint export and validate its positional contract."""

    metadata = pd.read_csv(metadata_path)
    if metadata.empty:
        raise ValueError(f"Reference metadata is empty: {metadata_path}")
    missing_columns = [column for column in (index_column, id_column) if column not in metadata]
    if missing_columns:
        raise ValueError(f"Reference metadata is missing required columns: {missing_columns}")

    numeric_index = pd.to_numeric(metadata[index_column], errors="raise").to_numpy()
    if not np.equal(numeric_index, numeric_index.astype(np.int64)).all():
        raise ValueError(f"Reference column {index_column!r} must contain integer indexes")
    metadata = metadata.assign(**{index_column: numeric_index.astype(np.int64)})
    metadata = metadata.sort_values(index_column, kind="stable").reset_index(drop=True)
    expected_index = np.arange(len(metadata), dtype=np.int64)
    if not np.array_equal(metadata[index_column].to_numpy(), expected_index):
        raise ValueError(
            f"Reference column {index_column!r} must contain every index from 0 to "
            f"{len(metadata) - 1} exactly once"
        )

    spectrum_ids = _decode_ids(metadata[id_column].to_numpy(), source="reference metadata")
    metadata[id_column] = spectrum_ids

    with np.load(fingerprints_path, allow_pickle=False) as arrays:
        if target_key not in arrays:
            raise ValueError(
                f"Reference array {target_key!r} is missing from {fingerprints_path}; "
                f"available keys: {arrays.files}"
            )
        targets = _validate_targets(arrays[target_key], len(metadata), str(fingerprints_path))

    return ReferenceBundle(
        metadata=metadata,
        spectrum_ids=spectrum_ids,
        targets=targets,
        fingerprint_bits=int(targets.shape[1]),
    )


def load_reference_predictions(
    fingerprints_path: str | Path,
    array_key: str,
    reference: ReferenceBundle,
) -> PredictionBundle:
    """Load a prediction array whose row order is locked by reference metadata."""

    with np.load(fingerprints_path, allow_pickle=False) as arrays:
        if array_key not in arrays:
            raise ValueError(
                f"Reference prediction {array_key!r} is missing from {fingerprints_path}; "
                f"available keys: {arrays.files}"
            )
        probabilities = _validate_probabilities(
            arrays[array_key],
            expected_rows=len(reference.metadata),
            fingerprint_bits=reference.fingerprint_bits,
            source=f"{fingerprints_path}:{array_key}",
        )
    return PredictionBundle(probabilities=probabilities, inference_seconds=None)


def load_prediction_bundle(
    path: str | Path,
    reference: ReferenceBundle,
    *,
    metadata_path: str | Path | None = None,
    metadata_id_column: str = "spec_name",
    metadata_index_column: str = "fingerprint_index",
    probabilities_key: str = STANDARD_PROBABILITIES_KEY,
    spectrum_ids_key: str = STANDARD_SPECTRUM_IDS_KEY,
    inference_seconds_key: str = STANDARD_INFERENCE_SECONDS_KEY,
) -> PredictionBundle:
    """Load candidate predictions and align them by explicit spectrum IDs.

    New prediction bundles should embed ``spectrum_ids`` in the NPZ. The
    optional companion metadata path supports historical FRIGID exports where
    IDs were stored separately; if both sources are present, they must agree.
    """

    metadata_ids = None
    if metadata_path is not None:
        metadata = pd.read_csv(metadata_path)
        if metadata_id_column not in metadata:
            raise ValueError(
                f"Prediction metadata {metadata_path} is missing {metadata_id_column!r}"
            )
        if metadata_index_column in metadata:
            numeric_index = pd.to_numeric(
                metadata[metadata_index_column], errors="raise"
            ).to_numpy()
            if not np.equal(numeric_index, numeric_index.astype(np.int64)).all():
                raise ValueError(
                    f"Prediction metadata column {metadata_index_column!r} must contain "
                    "integer indexes"
                )
            metadata = metadata.assign(
                **{metadata_index_column: numeric_index.astype(np.int64)}
            ).sort_values(metadata_index_column, kind="stable")
            expected_index = np.arange(len(metadata), dtype=np.int64)
            if not np.array_equal(
                metadata[metadata_index_column].to_numpy(), expected_index
            ):
                raise ValueError(
                    f"Prediction metadata column {metadata_index_column!r} must contain "
                    f"every index from 0 to {len(metadata) - 1} exactly once"
                )
        metadata_ids = _decode_ids(
            metadata[metadata_id_column].to_numpy(), source=str(metadata_path)
        )

    with np.load(path, allow_pickle=False) as arrays:
        required = [probabilities_key]
        missing = [key for key in required if key not in arrays]
        if missing:
            raise ValueError(
                f"Prediction bundle {path} is missing keys {missing}; available keys: {arrays.files}"
            )

        if spectrum_ids_key in arrays:
            candidate_ids = _decode_ids(arrays[spectrum_ids_key], source=str(path))
            if metadata_ids is not None and not np.array_equal(candidate_ids, metadata_ids):
                raise ValueError(
                    f"Embedded IDs in {path} do not match companion metadata {metadata_path}"
                )
        elif metadata_ids is not None:
            candidate_ids = metadata_ids
        else:
            raise ValueError(
                f"Prediction bundle {path} has no {spectrum_ids_key!r}; provide a companion "
                "metadata file instead"
            )
        probabilities = _validate_probabilities(
            arrays[probabilities_key],
            expected_rows=len(candidate_ids),
            fingerprint_bits=reference.fingerprint_bits,
            source=f"{path}:{probabilities_key}",
        )

        reference_set = set(reference.spectrum_ids.tolist())
        candidate_set = set(candidate_ids.tolist())
        missing_ids = reference_set - candidate_set
        extra_ids = candidate_set - reference_set
        if missing_ids or extra_ids:
            raise ValueError(
                f"Prediction IDs in {path} do not match the reference: "
                f"missing={sorted(missing_ids)[:5]}, extra={sorted(extra_ids)[:5]}"
            )

        row_by_id = {spectrum_id: row for row, spectrum_id in enumerate(candidate_ids.tolist())}
        order = np.asarray([row_by_id[spectrum_id] for spectrum_id in reference.spectrum_ids])
        probabilities = probabilities[order]

        inference_seconds = None
        if inference_seconds_key in arrays:
            inference_seconds = np.asarray(arrays[inference_seconds_key], dtype=np.float64)
            if inference_seconds.shape != (len(candidate_ids),):
                raise ValueError(
                    f"{path}:{inference_seconds_key} has shape {inference_seconds.shape}; "
                    f"expected {(len(candidate_ids),)}"
                )
            if not np.isfinite(inference_seconds).all() or np.any(inference_seconds < 0.0):
                raise ValueError(f"{path}:{inference_seconds_key} must be finite and non-negative")
            inference_seconds = inference_seconds[order]

    return PredictionBundle(
        probabilities=probabilities,
        inference_seconds=inference_seconds,
    )


def compute_per_spectrum_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    *,
    chunk_size: int = 2048,
) -> pd.DataFrame:
    """Compute row-level fingerprint metrics with bounded intermediate memory."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Threshold must be in [0, 1], got {threshold}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    targets = np.asarray(targets, dtype=bool)
    if targets.ndim != 2:
        raise ValueError(f"Targets must be two-dimensional, got {targets.shape}")
    probabilities = _validate_probabilities(
        probabilities,
        expected_rows=len(targets),
        fingerprint_bits=targets.shape[1],
        source="in-memory predictions",
    )
    if targets.shape != probabilities.shape:
        raise ValueError(
            f"Prediction shape {probabilities.shape} does not match targets {targets.shape}"
        )

    columns: dict[str, list[np.ndarray]] = {
        "fingerprint_tanimoto": [],
        "soft_tanimoto": [],
        "bce": [],
        "bit_accuracy": [],
        "bit_precision": [],
        "bit_recall": [],
        "bit_f1": [],
        "predicted_active_bits": [],
        "target_active_bits": [],
        "false_positive_bits": [],
        "false_negative_bits": [],
    }
    epsilon = np.float32(1e-7)

    for start in range(0, len(probabilities), chunk_size):
        stop = min(start + chunk_size, len(probabilities))
        probs = probabilities[start:stop]
        target = targets[start:stop]
        predicted = probs >= threshold

        true_positive = np.logical_and(predicted, target).sum(axis=1)
        false_positive = np.logical_and(predicted, ~target).sum(axis=1)
        false_negative = np.logical_and(~predicted, target).sum(axis=1)
        intersection = true_positive
        union = np.logical_or(predicted, target).sum(axis=1)

        tanimoto = np.divide(
            intersection,
            union,
            out=np.zeros(len(probs), dtype=np.float64),
            where=union > 0,
        )
        precision = np.divide(
            true_positive,
            true_positive + false_positive,
            out=np.zeros(len(probs), dtype=np.float64),
            where=(true_positive + false_positive) > 0,
        )
        recall = np.divide(
            true_positive,
            true_positive + false_negative,
            out=np.zeros(len(probs), dtype=np.float64),
            where=(true_positive + false_negative) > 0,
        )
        f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros(len(probs), dtype=np.float64),
            where=(precision + recall) > 0,
        )

        target_float = target.astype(np.float32, copy=False)
        soft_intersection = (probs * target_float).sum(axis=1, dtype=np.float64)
        soft_union = (probs + target_float - probs * target_float).sum(
            axis=1, dtype=np.float64
        )
        soft_tanimoto = np.divide(
            soft_intersection,
            soft_union,
            out=np.zeros(len(probs), dtype=np.float64),
            where=soft_union > 0,
        )
        clipped = np.clip(probs, epsilon, 1.0 - epsilon)
        bce = -(
            target_float * np.log(clipped) + (1.0 - target_float) * np.log(1.0 - clipped)
        ).mean(axis=1, dtype=np.float64)

        columns["fingerprint_tanimoto"].append(tanimoto)
        columns["soft_tanimoto"].append(soft_tanimoto)
        columns["bce"].append(bce)
        columns["bit_accuracy"].append((predicted == target).mean(axis=1))
        columns["bit_precision"].append(precision)
        columns["bit_recall"].append(recall)
        columns["bit_f1"].append(f1)
        columns["predicted_active_bits"].append(predicted.sum(axis=1))
        columns["target_active_bits"].append(target.sum(axis=1))
        columns["false_positive_bits"].append(false_positive)
        columns["false_negative_bits"].append(false_negative)

    return pd.DataFrame({name: np.concatenate(parts) for name, parts in columns.items()})


def aggregate_metrics(per_spectrum: pd.DataFrame) -> dict[str, float | int | None]:
    """Aggregate row-level metrics for one encoder."""

    result: dict[str, float | int | None] = {"rows": int(len(per_spectrum))}
    for column in (
        "fingerprint_tanimoto",
        "soft_tanimoto",
        "bce",
        "bit_accuracy",
        "bit_precision",
        "bit_recall",
        "bit_f1",
        "predicted_active_bits",
        "target_active_bits",
        "false_positive_bits",
        "false_negative_bits",
    ):
        result[f"mean_{column}"] = float(per_spectrum[column].mean())
    result["median_fingerprint_tanimoto"] = float(
        per_spectrum["fingerprint_tanimoto"].median()
    )

    if "inference_seconds" in per_spectrum:
        latency = per_spectrum["inference_seconds"].dropna().to_numpy(dtype=np.float64)
        result["mean_inference_seconds"] = float(latency.mean()) if len(latency) else None
        result["p95_inference_seconds"] = (
            float(np.quantile(latency, 0.95)) if len(latency) else None
        )
    else:
        result["mean_inference_seconds"] = None
        result["p95_inference_seconds"] = None
    return result


def paired_bootstrap_mean_ci(
    deltas: np.ndarray,
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
    cluster_ids: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return a deterministic paired bootstrap interval for a mean delta.

    When ``cluster_ids`` are supplied, complete molecule clusters are resampled
    and the statistic remains spectrum-weighted within each bootstrap sample.
    """

    deltas = np.asarray(deltas, dtype=np.float64)
    if deltas.ndim != 1 or not len(deltas):
        raise ValueError("Paired bootstrap requires a non-empty one-dimensional array")
    if samples <= 0:
        raise ValueError(f"Bootstrap samples must be positive, got {samples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"Confidence must be in (0, 1), got {confidence}")

    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    if cluster_ids is None:
        for index in range(samples):
            sample = rng.integers(0, len(deltas), size=len(deltas))
            means[index] = deltas[sample].mean()
    else:
        cluster_ids = np.asarray(cluster_ids).astype(str)
        if cluster_ids.shape != deltas.shape:
            raise ValueError(
                f"Cluster IDs have shape {cluster_ids.shape}; expected {deltas.shape}"
            )
        cluster_codes, unique_clusters = pd.factorize(cluster_ids, sort=True)
        if np.any(cluster_codes < 0):
            raise ValueError("Cluster IDs must not be empty or missing")
        cluster_sums = np.bincount(cluster_codes, weights=deltas)
        cluster_counts = np.bincount(cluster_codes)
        for index in range(samples):
            sample = rng.integers(0, len(unique_clusters), size=len(unique_clusters))
            means[index] = cluster_sums[sample].sum() / cluster_counts[sample].sum()
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def normalize_structure_identifier(value: object) -> str:
    """Normalize InChIKey-like identifiers to the structure-connectivity block."""

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text.split("-", maxsplit=1)[0].upper()


def structure_identifiers(metadata: pd.DataFrame) -> np.ndarray:
    """Return normalized structure clusters from a recognized InChIKey column."""

    candidates = (
        "inchi_key_first_block",
        "inchikey_first_block",
        "inchi_key",
        "inchikey",
    )
    column = next((candidate for candidate in candidates if candidate in metadata), None)
    if column is None:
        raise ValueError(
            "Reference metadata needs an InChIKey column for molecule-balanced metrics"
        )
    identifiers = metadata[column].map(normalize_structure_identifier).to_numpy(dtype=str)
    if np.any(identifiers == ""):
        raise ValueError(f"Reference metadata column {column!r} contains empty InChIKeys")
    return identifiers


def deterministic_cluster_partitions(
    metadata: pd.DataFrame,
    *,
    calibration_fraction: float,
    seed: int,
) -> np.ndarray:
    """Assign whole structure clusters to deterministic calibration/evaluation sets."""

    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError(
            f"Calibration fraction must be in (0, 1), got {calibration_fraction}"
        )
    cluster_ids = structure_identifiers(metadata)
    unique_clusters = sorted(set(cluster_ids.tolist()))
    if len(unique_clusters) < 2:
        raise ValueError("At least two structure clusters are required for partitioning")

    calibration_clusters = int(round(len(unique_clusters) * calibration_fraction))
    calibration_clusters = min(max(calibration_clusters, 1), len(unique_clusters) - 1)
    ranked_clusters = sorted(
        unique_clusters,
        key=lambda cluster: hashlib.sha256(
            f"{seed}\0{cluster}".encode("utf-8")
        ).digest(),
    )
    calibration_set = set(ranked_clusters[:calibration_clusters])
    return np.asarray(
        [
            "calibration" if cluster_id in calibration_set else "evaluation"
            for cluster_id in cluster_ids
        ],
        dtype=str,
    )


def select_reference_bundle(
    reference: ReferenceBundle,
    selection: pd.DataFrame,
    *,
    id_column: str,
    partition: str | None = None,
    partition_column: str = "benchmark_partition",
) -> tuple[ReferenceBundle, np.ndarray]:
    """Select reference rows by ID while preserving canonical reference order."""

    if id_column not in selection:
        raise ValueError(f"Selection manifest is missing ID column {id_column!r}")
    selection_ids = _decode_ids(selection[id_column].to_numpy(), source="selection manifest")
    selection = selection.copy()
    selection[id_column] = selection_ids
    unknown_ids = set(selection_ids.tolist()) - set(reference.spectrum_ids.tolist())
    if unknown_ids:
        raise ValueError(
            f"Selection manifest contains IDs absent from the reference: "
            f"{sorted(unknown_ids)[:5]}"
        )

    if partition is not None:
        if partition_column not in selection:
            raise ValueError(
                f"Selection manifest is missing partition column {partition_column!r}"
            )
        selection = selection.loc[selection[partition_column].astype(str).eq(partition)]
        if selection.empty:
            raise ValueError(f"Selection partition {partition!r} is empty")

    selected_ids = set(selection[id_column].tolist())
    positions = np.flatnonzero(
        np.asarray([spectrum_id in selected_ids for spectrum_id in reference.spectrum_ids])
    )
    if len(positions) != len(selected_ids):
        raise ValueError("Selection manifest could not be aligned to every requested ID")

    metadata = reference.metadata.iloc[positions].copy().reset_index(drop=True)
    if "fingerprint_index" in metadata:
        metadata.insert(
            metadata.columns.get_loc("fingerprint_index") + 1,
            "source_fingerprint_index",
            metadata["fingerprint_index"].to_numpy(),
        )
        metadata["fingerprint_index"] = np.arange(len(metadata), dtype=np.int64)
    if partition is not None:
        metadata[partition_column] = partition

    selected = ReferenceBundle(
        metadata=metadata,
        spectrum_ids=reference.spectrum_ids[positions],
        targets=reference.targets[positions],
        fingerprint_bits=reference.fingerprint_bits,
    )
    return selected, positions


def load_training_identifiers(path: str | Path) -> set[str]:
    """Load one identifier per line or a recognized identifier column from CSV."""

    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        candidates = (
            "inchi_key_first_block",
            "inchikey_first_block",
            "inchi_key",
            "inchikey",
        )
        column = next((candidate for candidate in candidates if candidate in frame), None)
        if column is None:
            raise ValueError(
                f"Training identifier CSV {path} needs one of these columns: {candidates}"
            )
        values: Iterable[object] = frame[column]
    else:
        values = path.read_text().splitlines()
    identifiers = {
        identifier
        for value in values
        if (identifier := normalize_structure_identifier(value))
    }
    if not identifiers:
        raise ValueError(f"Training identifier file {path} contains no usable identifiers")
    return identifiers


def training_overlap(
    metadata: pd.DataFrame,
    training_identifiers: set[str],
) -> tuple[int, float]:
    """Count evaluation structures present in a model's declared training set."""

    evaluation = pd.Series(structure_identifiers(metadata))
    overlaps = evaluation.isin(training_identifiers)
    return int(overlaps.sum()), float(overlaps.mean())
