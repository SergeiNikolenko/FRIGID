"""Target-blind fusion of frozen ranking and forward-spectrum scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from frigid.rankloop_inference import candidate_identity_sha256


def _zscore(values: np.ndarray) -> np.ndarray:
    standard_deviation = float(values.std())
    if standard_deviation <= 1e-8:
        return np.zeros_like(values, dtype=np.float64)
    return (values - values.mean()) / standard_deviation


def _rank_score(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values), dtype=np.int64)
    denominator = max(len(values) - 1, 1)
    return -ranks.astype(np.float64) / denominator


def fuse_forward_consistency_scores(
    frame: pd.DataFrame,
    *,
    alpha: float,
    normalization: str,
    mode: str,
    contradiction_quantile: float = 0.25,
) -> pd.DataFrame:
    """Rerank candidates while preserving candidate identity and failures."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between zero and one.")
    if normalization not in {"zscore", "rank"}:
        raise ValueError(f"Unsupported score normalization: {normalization!r}")
    if mode not in {"blend", "contradiction"}:
        raise ValueError(f"Unsupported forward fusion mode: {mode!r}")
    if not 0.0 < contradiction_quantile < 1.0:
        raise ValueError("contradiction_quantile must be between zero and one.")
    required = {
        "query_spec_name",
        "candidate_smiles",
        "rank",
        "tanimoto_to_mist",
        "forward_score",
    }
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"Forward fusion input is missing columns: {missing}")

    input_identity = candidate_identity_sha256(frame)
    ranked_parts: list[pd.DataFrame] = []
    for _query_name, query_rows in frame.groupby("query_spec_name", sort=False):
        rows = query_rows.copy()
        reference = pd.to_numeric(
            rows["tanimoto_to_mist"], errors="raise"
        ).to_numpy(dtype=np.float64)
        forward = pd.to_numeric(rows["forward_score"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        complete = bool(np.isfinite(forward).all())
        nondegenerate = complete and float(forward.max() - forward.min()) > 1e-8
        rows["forward_fallback"] = not nondegenerate

        if normalization == "zscore":
            reference_score = _zscore(reference)
            forward_score = _zscore(forward) if nondegenerate else np.zeros_like(reference)
        else:
            reference_score = _rank_score(reference)
            forward_score = _rank_score(forward) if nondegenerate else np.zeros_like(reference)

        if not nondegenerate:
            fused = reference_score
        elif mode == "blend":
            fused = (1.0 - alpha) * reference_score + alpha * forward_score
        else:
            cutoff = float(np.quantile(forward_score, contradiction_quantile))
            penalty = np.maximum(cutoff - forward_score, 0.0)
            if float(penalty.max()) > 0:
                penalty = penalty / penalty.max()
            fused = reference_score - alpha * penalty

        rows["forward_fusion_score"] = fused
        rows = rows.sort_values(
            ["forward_fusion_score", "rank"],
            ascending=[False, True],
            kind="mergesort",
        )
        rows["rankloop_rank"] = np.arange(1, len(rows) + 1, dtype=np.int64)
        ranked_parts.append(rows)

    ranked = pd.concat(ranked_parts, ignore_index=True)
    if candidate_identity_sha256(ranked) != input_identity:
        raise AssertionError("Forward fusion changed the frozen candidate pool.")
    return ranked
