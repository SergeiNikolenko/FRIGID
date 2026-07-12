import numpy as np
import pandas as pd

from frigid.forward_consistency import fuse_forward_consistency_scores
from frigid.rankloop_inference import candidate_identity_sha256


def _frame(forward_scores=(0.1, 0.9, 0.2)):
    return pd.DataFrame(
        {
            "query_spec_name": ["query-1"] * 3,
            "candidate_smiles": ["CC", "CCC", "CCCC"],
            "rank": [1, 2, 3],
            "tanimoto_to_mist": [0.9, 0.8, 0.7],
            "forward_score": list(forward_scores),
        }
    )


def test_forward_blend_can_promote_consistent_candidate_without_identity_change():
    frame = _frame()
    expected_identity = candidate_identity_sha256(frame)

    ranked = fuse_forward_consistency_scores(
        frame,
        alpha=0.8,
        normalization="zscore",
        mode="blend",
    )

    assert ranked.iloc[0]["candidate_smiles"] == "CCC"
    assert ranked["rankloop_rank"].tolist() == [1, 2, 3]
    assert candidate_identity_sha256(ranked) == expected_identity


def test_forward_contradiction_penalizes_only_low_forward_tail():
    ranked = fuse_forward_consistency_scores(
        _frame(),
        alpha=1.0,
        normalization="rank",
        mode="contradiction",
        contradiction_quantile=0.5,
    )

    assert ranked.iloc[0]["candidate_smiles"] == "CCC"
    assert ranked.iloc[-1]["candidate_smiles"] == "CCCC"


def test_missing_forward_score_falls_back_to_reference_order():
    ranked = fuse_forward_consistency_scores(
        _frame((0.1, np.nan, 0.2)),
        alpha=1.0,
        normalization="zscore",
        mode="blend",
    )

    assert ranked["candidate_smiles"].tolist() == ["CC", "CCC", "CCCC"]
    assert ranked["forward_fallback"].all()


def test_degenerate_forward_scores_fall_back_to_reference_order():
    ranked = fuse_forward_consistency_scores(
        _frame((0.5, 0.5, 0.5)),
        alpha=1.0,
        normalization="rank",
        mode="blend",
    )

    assert ranked["candidate_smiles"].tolist() == ["CC", "CCC", "CCCC"]
    assert ranked["forward_fallback"].all()
