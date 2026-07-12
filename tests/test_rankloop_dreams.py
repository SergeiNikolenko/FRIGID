import numpy as np
import pytest

from frigid.rankloop_dreams import (
    build_dreams_mgf_entry,
    merge_spectrum_peaks,
    validate_dreams_embeddings,
)


def test_merge_spectrum_peaks_preserves_all_rows_and_sorts_by_mz():
    peaks = merge_spectrum_peaks(
        [
            ("energy_1", np.array([[200.0, 0.2], [50.0, 1.0]])),
            ("energy_2", np.array([[100.0, 0.5]])),
        ]
    )

    np.testing.assert_allclose(
        peaks,
        np.array([[50.0, 1.0], [100.0, 0.5], [200.0, 0.2]]),
    )


def test_build_dreams_mgf_entry_contains_required_metadata_and_peaks():
    entry = build_dreams_mgf_entry(
        spec_name="spec-1",
        formula="C2H6O",
        precursor_mz=47.0491,
        peaks=np.array([[31.0, 1.0], [45.0, 0.2]]),
    )

    assert entry.splitlines() == [
        "BEGIN IONS",
        "TITLE=spec-1",
        "NAME=spec-1",
        "SCANS=spec-1",
        "PEPMASS=47.0491",
        "FORMULA=C2H6O",
        "31 1",
        "45 0.2",
        "END IONS",
    ]


def test_validate_dreams_embeddings_rejects_row_mismatch():
    with pytest.raises(ValueError, match="row count"):
        validate_dreams_embeddings(
            np.zeros((2, 1024), dtype=np.float32),
            expected_rows=3,
            expected_dimension=1024,
        )


def test_validate_dreams_embeddings_returns_float32_matrix():
    matrix = validate_dreams_embeddings(
        np.ones((3, 4), dtype=np.float64),
        expected_rows=3,
        expected_dimension=4,
    )

    assert matrix.dtype == np.float32
    assert matrix.shape == (3, 4)
