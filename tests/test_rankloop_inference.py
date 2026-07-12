from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from frigid.rankloop_inference import (
    candidate_identity_sha256,
    load_inference_candidate_frame,
    score_candidate_frame,
)
from frigid.rankloop_model import MoleculeEmbeddingTable, SpectrumEmbeddingTable


class _IdentityDualEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(0.0))

    def encode_spectra(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(values, dim=-1)

    def encode_molecules(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(values, dim=-1)


def _write_candidates(tmp_path: Path, **extra_columns: list[str]) -> Path:
    frame = pd.DataFrame(
        {
            "spec_name": ["q1", "q1", "q2", "q2"],
            "smiles": ["C", "N", "C", "N"],
            "source_name": ["a", "b", "a", "b"],
            **extra_columns,
        }
    )
    path = tmp_path / "candidates.csv"
    frame.to_csv(path, index=False)
    return path


def test_inference_rejects_target_columns(tmp_path: Path):
    path = _write_candidates(tmp_path, target_smiles=["C"] * 4)

    with pytest.raises(ValueError, match="forbidden target-derived"):
        load_inference_candidate_frame(path)


def test_reranking_is_target_blind_and_preserves_candidate_identity(tmp_path: Path):
    candidates = load_inference_candidate_frame(_write_candidates(tmp_path))
    identity_before = candidate_identity_sha256(candidates)
    spectrum_table = SpectrumEmbeddingTable(
        by_spec_name={
            "q1": np.asarray([1.0, 0.0], dtype=np.float32),
            "q2": np.asarray([0.0, 1.0], dtype=np.float32),
        },
        dimension=2,
    )
    molecule_table = MoleculeEmbeddingTable(
        by_smiles={
            "C": np.asarray([1.0, 0.0], dtype=np.float32),
            "N": np.asarray([0.0, 1.0], dtype=np.float32),
        },
        dimension=2,
    )

    ranked = score_candidate_frame(
        candidates,
        spectrum_table,
        molecule_table,
        _IdentityDualEncoder(),
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert candidate_identity_sha256(ranked) == identity_before
    winners = ranked.loc[ranked["rankloop_rank"] == 1]
    assert dict(zip(winners["query_spec_name"], winners["candidate_smiles"])) == {
        "q1": "C",
        "q2": "N",
    }
    assert ranked.groupby("query_spec_name")["rankloop_rank"].max().eq(2).all()
