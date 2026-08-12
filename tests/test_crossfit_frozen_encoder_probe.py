"""Fold assignment for the out-of-fold fingerprint probe.

The whole point of the cross-fit is that no row is ever predicted by a probe
that saw its structure, so the fold map is the part worth testing: a
connectivity block must live in exactly one fold, and every row must land in
one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from crossfit_frozen_encoder_probe import assign_folds  # noqa: E402


def _structures(count: int, repeats: int) -> np.ndarray:
    return np.asarray(
        [f"BLOCK{index:04d}" for index in range(count) for _ in range(repeats)],
        dtype=str,
    )


def test_a_connectivity_block_never_spans_two_folds():
    structures = _structures(50, 3)

    folds = assign_folds(structures, folds=5, seed=42)

    assert folds.shape == structures.shape
    for structure in set(structures.tolist()):
        assigned = set(folds[structures == structure].tolist())
        assert len(assigned) == 1, structure


def test_every_fold_is_used_and_every_row_is_covered():
    structures = _structures(50, 2)

    folds = assign_folds(structures, folds=5, seed=42)

    assert set(folds.tolist()) == set(range(5))
    assert np.all(folds >= 0)


def test_the_map_is_stable_for_a_seed_and_moves_with_it():
    structures = _structures(30, 1)

    first = assign_folds(structures, folds=3, seed=42)
    again = assign_folds(structures, folds=3, seed=42)
    other = assign_folds(structures, folds=3, seed=7)

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_fewer_blocks_than_folds_is_refused():
    with pytest.raises(ValueError):
        assign_folds(_structures(3, 4), folds=5, seed=42)
    with pytest.raises(ValueError):
        assign_folds(_structures(10, 1), folds=1, seed=42)
