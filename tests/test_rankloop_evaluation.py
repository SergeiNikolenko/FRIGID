from __future__ import annotations

import pandas as pd
import pytest

from frigid.rankloop_evaluation import (
    evaluate_ranked_candidates,
    validate_identical_candidate_pool,
)


def _ranked(rank_column: str, ranks: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_spec_name": ["q1", "q1", "q2", "q2"],
            "candidate_smiles": ["C", "N", "C", "N"],
            rank_column: ranks,
        }
    )


def test_rankloop_evaluation_preserves_recall_and_improves_ranking():
    reference = _ranked("rank", [2, 1, 1, 2])
    candidate = _ranked("rankloop_rank", [1, 2, 2, 1])
    identity_hash = validate_identical_candidate_pool(reference, candidate)
    targets = pd.DataFrame(
        {
            "spec_name": ["q1", "q2"],
            "target_smiles": ["C", "N"],
            "target_inchi_key": ["VNWKTOKETHGBQD", "QGZKDVFQNNGYKY"],
        }
    )

    reference_metrics = evaluate_ranked_candidates(
        reference,
        targets,
        rank_column="rank",
        method_name="reference",
        fingerprint_bits=128,
    )
    candidate_metrics = evaluate_ranked_candidates(
        candidate,
        targets,
        rank_column="rankloop_rank",
        method_name="rankloop",
        fingerprint_bits=128,
    )

    assert len(identity_hash) == 64
    assert reference_metrics["candidate_recall_exact"].eq(1.0).all()
    assert candidate_metrics["candidate_recall_exact"].eq(1.0).all()
    assert reference_metrics["exact_match_top1"].eq(0.0).all()
    assert candidate_metrics["exact_match_top1"].eq(1.0).all()
    assert candidate_metrics["ranking_regret_top1"].eq(0.0).all()


def test_rankloop_evaluation_rejects_candidate_pool_changes():
    reference = _ranked("rank", [1, 2, 1, 2])
    candidate = _ranked("rankloop_rank", [1, 2, 1, 2]).iloc[:-1].copy()

    with pytest.raises(ValueError, match="differs from the frozen reference"):
        validate_identical_candidate_pool(reference, candidate)
