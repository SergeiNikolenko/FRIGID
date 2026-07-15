from __future__ import annotations

import numpy as np
import pytest

from frigid.frozen_probe import (
    deterministic_group_holdout,
    global_positive_weight,
    load_embedding_bundle,
)


def test_load_embedding_bundle_validates_and_normalizes(tmp_path):
    path = tmp_path / "bundle.npz"
    np.savez_compressed(
        path,
        embeddings=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        ground_truth=np.asarray([[1, 0, 0, 1], [0, 1, 0, 0]], dtype=np.uint8),
        spectrum_ids=np.asarray(["s1", "s2"]),
        inchikeys=np.asarray(["ABCDEFGHIJKLMN-AA", "ZYXWVUTSRQPONM-BB"]),
        inference_seconds=np.asarray(0.4),
    )

    bundle = load_embedding_bundle(str(path), fingerprint_bits=4)

    assert bundle.embeddings.shape == (2, 2)
    assert bundle.targets.dtype == np.uint8
    assert bundle.structure_ids.tolist() == ["ABCDEFGHIJKLMN", "ZYXWVUTSRQPONM"]
    assert bundle.inference_seconds.tolist() == pytest.approx([0.2, 0.2])


def test_load_embedding_bundle_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "bundle.npz"
    np.savez_compressed(
        path,
        embeddings=np.ones((2, 2), dtype=np.float32),
        ground_truth=np.zeros((2, 4), dtype=np.uint8),
        spectrum_ids=np.asarray(["s1", "s1"]),
        inchikeys=np.asarray(["A", "B"]),
    )

    with pytest.raises(ValueError, match="duplicates"):
        load_embedding_bundle(str(path), fingerprint_bits=4)


def test_group_holdout_keeps_structures_together_and_is_deterministic():
    structures = np.asarray(["A", "A", "B", "C", "C", "D"])

    train_a, validation_a = deterministic_group_holdout(
        structures, validation_fraction=0.5, seed=42
    )
    train_b, validation_b = deterministic_group_holdout(
        structures, validation_fraction=0.5, seed=42
    )

    assert np.array_equal(train_a, train_b)
    assert np.array_equal(validation_a, validation_b)
    assert set(structures[train_a]).isdisjoint(set(structures[validation_a]))


def test_global_positive_weight_counts_selected_rows():
    targets = np.asarray([[1, 0, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1]])

    assert global_positive_weight(targets, np.asarray([0, 1])) == pytest.approx(5 / 3)
