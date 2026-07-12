"""Target-blind fusion of frozen MIST and RankLoop candidate scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from frigid.rankloop_inference import candidate_identity_sha256


def fuse_rankloop_scores(
    frame: pd.DataFrame,
    *,
    alpha: float,
    normalization: str,
) -> pd.DataFrame:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between zero and one.")
    required = {
        "query_spec_name",
        "candidate_smiles",
        "rank",
        "tanimoto_to_mist",
        "rankloop_score",
        "rankloop_rank",
    }
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"RankLoop fusion input is missing columns: {missing}")
    if normalization not in {"zscore", "rank"}:
        raise ValueError(f"Unsupported score normalization: {normalization!r}")

    input_identity_hash = candidate_identity_sha256(frame)
    ranked_parts = []
    for _query_name, rows in frame.groupby("query_spec_name", sort=True):
        rows = rows.copy()
        if normalization == "zscore":
            mist = rows["tanimoto_to_mist"].to_numpy(dtype=np.float64)
            learned = rows["rankloop_score"].to_numpy(dtype=np.float64)
            mist_std = float(mist.std())
            learned_std = float(learned.std())
            mist = (mist - mist.mean()) / max(mist_std, 1e-8)
            learned = (learned - learned.mean()) / max(learned_std, 1e-8)
        else:
            denominator = max(len(rows) - 1, 1)
            mist = -(rows["rank"].to_numpy(dtype=np.float64) - 1.0) / denominator
            learned = (
                -(rows["rankloop_rank"].to_numpy(dtype=np.float64) - 1.0) / denominator
            )
        rows["rankloop_fusion_score"] = (1.0 - alpha) * mist + alpha * learned
        rows = rows.sort_values(
            ["rankloop_fusion_score", "rank"],
            ascending=[False, True],
            kind="mergesort",
        )
        rows["dual_encoder_rank"] = rows["rankloop_rank"].astype(int)
        rows["rankloop_rank"] = np.arange(1, len(rows) + 1, dtype=np.int64)
        ranked_parts.append(rows)
    ranked = pd.concat(ranked_parts, ignore_index=True)
    if candidate_identity_sha256(ranked) != input_identity_hash:
        raise AssertionError("RankLoop fusion changed the frozen candidate pool.")
    return ranked
