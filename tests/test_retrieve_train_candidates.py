import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "retrieve_train_candidates.py"
SPEC = importlib.util.spec_from_file_location("retrieve_train_candidates", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["retrieve_train_candidates"] = MODULE
SPEC.loader.exec_module(MODULE)


def test_split_isolation_selects_only_requested_query_splits():
    split_rows = [
        ("s_train_1", "train"),
        ("s_val_1", "val"),
        ("s_test_1", "test"),
        ("s_val_2", "val"),
    ]

    train_specs, query_specs = MODULE.resolve_split_assignments(
        split_rows,
        ("val",),
    )

    assert train_specs == ["s_train_1"]
    assert query_specs == ["s_val_1", "s_val_2"]


def test_leakage_rejection_default_fails_and_override_allows():
    candidates = [
        MODULE.MoleculeCandidate(
            spec_name="a",
            smiles="CC",
            inchi_key_first_block="AAA",
            formula="C2H6",
            fingerprint=np.array([1, 0, 1], dtype=np.float32),
        ),
        MODULE.MoleculeCandidate(
            spec_name="b",
            smiles="CO",
            inchi_key_first_block="BBB",
            formula="CH4O",
            fingerprint=np.array([0, 1, 1], dtype=np.float32),
        ),
    ]
    deduped = MODULE.deduplicate_by_inchikey_first_block(candidates)

    query_blocks = ["AAA", "CCC"]
    with pytest.raises(ValueError, match="overlap"):
        MODULE.validate_query_leakage(query_blocks, {c.inchi_key_first_block for c in deduped}, False)

    # Explicit override must be opt-in and permit processing.
    MODULE.validate_query_leakage(query_blocks, {c.inchi_key_first_block for c in deduped}, True)


def test_exported_queries_are_restricted_to_requested_splits():
    assert MODULE.select_exported_query_specs(
        ["s_test_2", "s_test_1"],
        {"s_test_1", "s_test_2", "s_test_3"},
    ) == ["s_test_2", "s_test_1"]

    with pytest.raises(ValueError, match="outside --query-splits"):
        MODULE.select_exported_query_specs(
            ["s_train_1"],
            {"s_test_1", "s_test_2"},
        )


def test_formula_first_ranking_and_global_fallback():
    candidates = [
        MODULE.MoleculeCandidate(
            spec_name="match_low",
            smiles="O=O",
            inchi_key_first_block="M1",
            formula="C2H4",
            fingerprint=np.array([1, 0, 0, 0], dtype=np.float32),
        ),
        MODULE.MoleculeCandidate(
            spec_name="match_high",
            smiles="CC",
            inchi_key_first_block="M2",
            formula="C2H4",
            fingerprint=np.array([0, 1, 0, 0], dtype=np.float32),
        ),
        MODULE.MoleculeCandidate(
            spec_name="non_match",
            smiles="N",
            inchi_key_first_block="N1",
            formula="N",
            fingerprint=np.array([1, 1, 1, 1], dtype=np.float32),
        ),
    ]

    query_fp = np.array([1, 1, 0, 0], dtype=np.float32)
    ranked = MODULE.rank_train_candidates(query_fp, "C2H4", candidates, top_k=3)

    assert [entry.candidate.spec_name for entry in ranked[:2]] == ["match_low", "match_high"]
    assert ranked[2].candidate.spec_name == "non_match"


def test_matrix_backend_matches_legacy_formula_and_tie_order():
    rng = np.random.default_rng(17)
    candidates = []
    for index in range(24):
        fingerprint = rng.integers(0, 2, size=64).astype(np.float32)
        if index == 1:
            fingerprint = candidates[0].fingerprint.copy()
        candidates.append(
            MODULE.MoleculeCandidate(
                spec_name=f"train_{index}",
                smiles=f"C{index}",
                inchi_key_first_block=f"KEY{index}",
                formula="MATCH" if index < 4 else f"F{index}",
                fingerprint=fingerprint,
            )
        )
    query_fingerprints = np.stack(
        [
            candidates[0].fingerprint,
            rng.integers(0, 2, size=64).astype(np.float32),
        ]
    )
    formulas = ["MATCH", "ABSENT"]
    matrix = MODULE.MatrixTrainCandidateIndex(candidates, torch.device("cpu"))

    observed = matrix.rank_batch(query_fingerprints, formulas, top_k=10)
    expected = [
        MODULE.rank_train_candidates(query_fp, formula, candidates, top_k=10)
        for query_fp, formula in zip(query_fingerprints, formulas)
    ]

    for observed_query, expected_query in zip(observed, expected):
        assert [row.candidate.spec_name for row in observed_query] == [
            row.candidate.spec_name for row in expected_query
        ]
        assert [row.formula_match for row in observed_query] == [
            row.formula_match for row in expected_query
        ]
        assert np.allclose(
            [row.tanimoto for row in observed_query],
            [row.tanimoto for row in expected_query],
            rtol=0,
            atol=0,
        )


def test_deduplicate_by_inchi_key_first_block_keeps_first_occurrence():
    candidates = [
        MODULE.MoleculeCandidate(
            spec_name="dup_a",
            smiles="CC",
            inchi_key_first_block="AAA",
            formula="C2H6",
            fingerprint=np.array([1, 0], dtype=np.float32),
        ),
        MODULE.MoleculeCandidate(
            spec_name="dup_b",
            smiles="CCO",
            inchi_key_first_block="AAA",
            formula="C2H6O",
            fingerprint=np.array([1, 1], dtype=np.float32),
        ),
        MODULE.MoleculeCandidate(
            spec_name="unique",
            smiles="N",
            inchi_key_first_block="BBB",
            formula="NH3",
            fingerprint=np.array([0, 1], dtype=np.float32),
        ),
    ]

    deduped = MODULE.deduplicate_by_inchikey_first_block(candidates)

    assert [c.spec_name for c in deduped] == ["dup_a", "unique"]


def test_metrics_are_computed_for_ranked_predictions():
    ranked = [
        MODULE.RankedCandidate(
            rank=1,
            candidate=MODULE.MoleculeCandidate(
                spec_name="s1",
                smiles="CC",
                inchi_key_first_block="A",
                formula="C2H6",
                fingerprint=np.array([1, 1, 0, 0], dtype=np.float32),
            ),
            tanimoto=0.5,
            formula_match=True,
        ),
        MODULE.RankedCandidate(
            rank=2,
            candidate=MODULE.MoleculeCandidate(
                spec_name="s2",
                smiles="CO",
                inchi_key_first_block="B",
                formula="CH4O",
                fingerprint=np.array([0, 1, 1, 0], dtype=np.float32),
            ),
            tanimoto=0.333,
            formula_match=False,
        ),
    ]

    target_fp = np.array([1, 0, 0, 0], dtype=np.float32)
    metrics = MODULE.evaluate_ranked_predictions("B", target_fp, ranked, top_k=2)

    assert metrics["exact_match_top1"] == 0.0
    assert metrics["exact_match_top10"] == 1.0
    assert metrics["tanimoto_top1"] == 0.5
    assert metrics["tanimoto_top10"] == 0.5
    assert metrics["formula_match_count"] == 1.0
