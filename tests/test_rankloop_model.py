from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from frigid.rankloop_model import (
    MorganRankLoopDualEncoder,
    RankLoopCorpusDataset,
    RankLoopDenseCorpusDataset,
    collate_rankloop_lists,
    compute_rankloop_loss,
    load_spectrum_embedding_table,
    load_molecule_embedding_table,
    positive_set_nll,
)


def _write_embeddings(tmp_path: Path) -> tuple[Path, Path]:
    metadata = tmp_path / "metadata.csv"
    embeddings = tmp_path / "embeddings.npz"
    pd.DataFrame(
        [
            {"embedding_index": 0, "spec_name": "q1"},
            {"embedding_index": 1, "spec_name": "q2"},
        ]
    ).to_csv(metadata, index=False)
    np.savez_compressed(
        embeddings,
        spectrum_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        spec_names=np.array(["q1", "q2"]),
    )
    return metadata, embeddings


def _write_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus.csv"
    pd.DataFrame(
        [
            ("q1", "train", "A", "CCO", "A", 1),
            ("q1", "train", "A", "COC", "B", 0),
            ("q2", "development", "B", "COC", "B", 1),
            ("q2", "development", "B", "CCO", "A", 0),
        ],
        columns=(
            "query_spec_name",
            "rankloop_split",
            "query_inchi_key_first_block",
            "candidate_smiles",
            "candidate_inchi_key_first_block",
            "label",
        ),
    ).to_csv(corpus, index=False)
    return corpus


def test_embedding_contract_and_corpus_collation(tmp_path: Path):
    metadata, embeddings = _write_embeddings(tmp_path)
    corpus = _write_corpus(tmp_path)
    table = load_spectrum_embedding_table(metadata, embeddings)
    train = RankLoopCorpusDataset(corpus, table, partition="train", fingerprint_bits=64)
    development = RankLoopCorpusDataset(
        corpus,
        table,
        partition="development",
        fingerprint_bits=64,
    )
    batch = collate_rankloop_lists([train[0], development[0]])

    assert table.dimension == 2
    assert batch["spectrum_embeddings"].shape == (2, 2)
    assert batch["candidate_features"].shape == (2, 2, 64)
    assert batch["positive_mask"].sum(dim=1).tolist() == [1, 1]


def test_embedding_contract_rejects_misaligned_names(tmp_path: Path):
    metadata, embeddings = _write_embeddings(tmp_path)
    pd.DataFrame([{"spec_name": "q2"}, {"spec_name": "q1"}]).to_csv(
        metadata, index=False
    )

    with pytest.raises(ValueError, match="do not match metadata order"):
        load_spectrum_embedding_table(metadata, embeddings)


def test_precomputed_molecule_embeddings_use_the_same_corpus_contract(tmp_path: Path):
    spectrum_metadata, spectrum_embeddings = _write_embeddings(tmp_path)
    corpus = _write_corpus(tmp_path)
    molecule_metadata = tmp_path / "molecule_metadata.csv"
    molecule_embeddings = tmp_path / "molecule_embeddings.npz"
    pd.DataFrame([{"smiles": "CCO"}, {"smiles": "COC"}]).to_csv(
        molecule_metadata,
        index=False,
    )
    np.savez_compressed(
        molecule_embeddings,
        molecule_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        smiles=np.array(["CCO", "COC"]),
    )
    spectrum_table = load_spectrum_embedding_table(
        spectrum_metadata,
        spectrum_embeddings,
    )
    molecule_table = load_molecule_embedding_table(
        molecule_metadata,
        molecule_embeddings,
    )
    dataset = RankLoopDenseCorpusDataset(
        corpus,
        spectrum_table,
        molecule_table,
        partition="train",
    )
    batch = collate_rankloop_lists([dataset[0]])

    assert molecule_table.dimension == 2
    assert batch["candidate_features"].shape == (1, 2, 2)


def test_positive_set_nll_supports_multiple_positives():
    logits = torch.tensor([[3.0, 2.0, -1.0], [0.0, 1.0, 2.0]])
    positives = torch.tensor([[True, True, False], [False, False, True]])
    loss = positive_set_nll(logits, positives)

    assert torch.isfinite(loss)
    assert (
        loss.item()
        < positive_set_nll(
            logits, torch.tensor([[False, True, False], [False, True, False]])
        ).item()
    )


def test_dual_encoder_loss_is_finite_with_duplicate_positive_groups():
    torch.manual_seed(7)
    model = MorganRankLoopDualEncoder(
        spectrum_dimension=4,
        fingerprint_bits=8,
        embedding_dimension=4,
        hidden_dimension=8,
        dropout=0.0,
    )
    spectra = torch.randn(3, 4)
    candidates = torch.randn(3, 3, 8)
    candidate_mask = torch.ones(3, 3, dtype=torch.bool)
    positive_mask = torch.tensor(
        [[True, False, False], [False, True, False], [True, False, False]]
    )
    group_ids = torch.tensor([0, 1, 0])
    logits, spectrum_latent, candidate_latent = model(spectra, candidates)
    losses = compute_rankloop_loss(
        logits,
        spectrum_latent,
        candidate_latent,
        candidate_mask=candidate_mask,
        positive_mask=positive_mask,
        query_group_ids=group_ids,
        logit_scale=model.logit_scale.exp(),
    )

    assert all(torch.isfinite(value) for value in losses.values())
    assert 0.0 <= losses["top1_accuracy"].item() <= 1.0
