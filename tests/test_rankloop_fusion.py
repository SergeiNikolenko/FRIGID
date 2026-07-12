from __future__ import annotations

import pandas as pd

from frigid.rankloop_fusion import fuse_rankloop_scores
from frigid.rankloop_inference import candidate_identity_sha256


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_spec_name": ["q1", "q1", "q1"],
            "candidate_smiles": ["C", "N", "O"],
            "rank": [1, 2, 3],
            "tanimoto_to_mist": [0.9, 0.8, 0.7],
            "rankloop_score": [0.0, 1.0, 2.0],
            "rankloop_rank": [3, 2, 1],
        }
    )


def test_zero_weight_reproduces_frozen_mist_ranking():
    frame = _frame()
    identity = candidate_identity_sha256(frame)

    ranked = fuse_rankloop_scores(frame, alpha=0.0, normalization="zscore")

    assert ranked.sort_values("rankloop_rank")["candidate_smiles"].tolist() == [
        "C",
        "N",
        "O",
    ]
    assert candidate_identity_sha256(ranked) == identity


def test_rankloop_weight_can_change_ranking_without_changing_candidates():
    frame = _frame()

    ranked = fuse_rankloop_scores(frame, alpha=1.0, normalization="rank")

    assert ranked.sort_values("rankloop_rank")["candidate_smiles"].tolist() == [
        "O",
        "N",
        "C",
    ]
    assert ranked["dual_encoder_rank"].tolist() == [1, 2, 3]
