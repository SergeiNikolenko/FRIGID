from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frigid.encoder_benchmark import (
    compute_per_spectrum_metrics,
    load_prediction_bundle,
    load_reference_bundle,
    load_training_identifiers,
    paired_bootstrap_mean_ci,
    training_overlap,
)


def make_reference(tmp_path: Path):
    metadata_path = tmp_path / "metadata.csv"
    fingerprints_path = tmp_path / "fingerprints.npz"
    pd.DataFrame(
        {
            "fingerprint_index": [1, 0],
            "spec_name": ["spec-b", "spec-a"],
            "inchi_key_first_block": ["BBBBBBBBBBBBBB", "AAAAAAAAAAAAAA"],
        }
    ).to_csv(metadata_path, index=False)
    np.savez_compressed(
        fingerprints_path,
        ground_truth=np.asarray([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=np.float32),
        mist_probs=np.asarray([[0.9, 0.1, 0.8, 0.2], [0.6, 0.7, 0.1, 0.2]], dtype=np.float32),
    )
    return metadata_path, fingerprints_path


def test_reference_is_aligned_by_fingerprint_index(tmp_path):
    metadata_path, fingerprints_path = make_reference(tmp_path)
    reference = load_reference_bundle(metadata_path, fingerprints_path)

    assert reference.spectrum_ids.tolist() == ["spec-a", "spec-b"]
    assert reference.targets.tolist() == [[True, False, True, False], [False, True, False, False]]


def test_external_predictions_are_realigned_by_spectrum_id(tmp_path):
    metadata_path, fingerprints_path = make_reference(tmp_path)
    reference = load_reference_bundle(metadata_path, fingerprints_path)
    prediction_path = tmp_path / "candidate.npz"
    np.savez_compressed(
        prediction_path,
        spectrum_ids=np.asarray(["spec-b", "spec-a"]),
        probs=np.asarray([[0.1, 0.9, 0.1, 0.1], [0.9, 0.1, 0.8, 0.2]], dtype=np.float32),
        inference_seconds=np.asarray([2.0, 1.0]),
    )

    candidate = load_prediction_bundle(prediction_path, reference)

    assert candidate.probabilities[0].tolist() == pytest.approx([0.9, 0.1, 0.8, 0.2])
    assert candidate.inference_seconds.tolist() == [1.0, 2.0]


def test_external_predictions_require_exact_id_set(tmp_path):
    metadata_path, fingerprints_path = make_reference(tmp_path)
    reference = load_reference_bundle(metadata_path, fingerprints_path)
    prediction_path = tmp_path / "candidate.npz"
    np.savez_compressed(
        prediction_path,
        spectrum_ids=np.asarray(["spec-a", "spec-c"]),
        probs=np.zeros((2, 4), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="do not match the reference"):
        load_prediction_bundle(prediction_path, reference)


def test_historical_predictions_can_use_companion_metadata(tmp_path):
    metadata_path, fingerprints_path = make_reference(tmp_path)
    reference = load_reference_bundle(metadata_path, fingerprints_path)
    prediction_path = tmp_path / "candidate.npz"
    prediction_metadata_path = tmp_path / "candidate_metadata.csv"
    np.savez_compressed(
        prediction_path,
        probs=np.asarray([[0.9, 0.1, 0.8, 0.2], [0.1, 0.9, 0.1, 0.1]], dtype=np.float32),
    )
    pd.DataFrame(
        {
            "fingerprint_index": [1, 0],
            "spec_name": ["spec-b", "spec-a"],
        }
    ).to_csv(prediction_metadata_path, index=False)

    candidate = load_prediction_bundle(
        prediction_path,
        reference,
        metadata_path=prediction_metadata_path,
    )

    assert candidate.probabilities[0].tolist() == pytest.approx([0.9, 0.1, 0.8, 0.2])


def test_fingerprint_metrics_match_known_values():
    targets = np.asarray([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=bool)
    probabilities = np.asarray(
        [[0.9, 0.1, 0.8, 0.2], [0.6, 0.7, 0.1, 0.2]], dtype=np.float32
    )

    metrics = compute_per_spectrum_metrics(probabilities, targets, threshold=0.5, chunk_size=1)

    assert metrics["fingerprint_tanimoto"].tolist() == pytest.approx([1.0, 0.5])
    assert metrics["bit_precision"].tolist() == pytest.approx([1.0, 0.5])
    assert metrics["bit_recall"].tolist() == pytest.approx([1.0, 1.0])
    assert metrics["predicted_active_bits"].tolist() == [2, 2]
    assert metrics["false_positive_bits"].tolist() == [0, 1]
    assert metrics["false_negative_bits"].tolist() == [0, 0]
    assert np.isfinite(metrics["bce"]).all()


def test_fingerprint_metrics_reject_one_dimensional_targets():
    with pytest.raises(ValueError, match="Targets must be two-dimensional"):
        compute_per_spectrum_metrics(
            np.asarray([[0.1, 0.2]], dtype=np.float32),
            np.asarray([1, 0], dtype=bool),
            threshold=0.5,
        )


def test_paired_bootstrap_is_deterministic():
    deltas = np.asarray([0.1, 0.2, 0.3, 0.4])
    clusters = np.asarray(["a", "a", "b", "c"])

    first = paired_bootstrap_mean_ci(deltas, samples=100, seed=7, cluster_ids=clusters)
    second = paired_bootstrap_mean_ci(deltas, samples=100, seed=7, cluster_ids=clusters)

    assert first == second
    assert first[0] > 0.0


def test_training_overlap_uses_first_inchikey_block():
    metadata = pd.DataFrame(
        {"inchi_key": ["AAAAAAAAAAAAAA-UHFFFAOYSA-N", "BBBBBBBBBBBBBB-UHFFFAOYSA-N"]}
    )

    rows, rate = training_overlap(metadata, {"AAAAAAAAAAAAAA"})

    assert rows == 1
    assert rate == 0.5


def test_training_identifier_file_must_not_be_empty(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("\n")

    with pytest.raises(ValueError, match="no usable identifiers"):
        load_training_identifiers(path)
