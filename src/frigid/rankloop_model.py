"""Core data and model components for RankLoop dual-encoder experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from frigid.rankloop_corpus import molecule_record_from_smiles


REQUIRED_CORPUS_COLUMNS = frozenset(
    {
        "query_spec_name",
        "rankloop_split",
        "query_inchi_key_first_block",
        "candidate_smiles",
        "candidate_inchi_key_first_block",
        "label",
    }
)


@dataclass(frozen=True)
class SpectrumEmbeddingTable:
    by_spec_name: dict[str, np.ndarray]
    dimension: int


def load_spectrum_embedding_table(
    metadata_csv: str | Path,
    embeddings_npz: str | Path,
) -> SpectrumEmbeddingTable:
    metadata = pd.read_csv(metadata_csv, dtype=str).fillna("")
    if "spec_name" not in metadata.columns:
        raise ValueError("Spectrum embedding metadata requires spec_name.")
    if metadata["spec_name"].duplicated().any():
        duplicate = metadata.loc[metadata["spec_name"].duplicated(), "spec_name"].iloc[
            0
        ]
        raise ValueError(f"Duplicate spectrum embedding metadata row: {duplicate}")

    with np.load(embeddings_npz, allow_pickle=False) as archive:
        if "spectrum_embeddings" not in archive:
            raise ValueError("Embedding NPZ requires spectrum_embeddings.")
        embeddings = np.asarray(archive["spectrum_embeddings"], dtype=np.float32)
        archived_names = (
            [str(value) for value in archive["spec_names"]]
            if "spec_names" in archive
            else None
        )
    if embeddings.ndim != 2:
        raise ValueError("spectrum_embeddings must be a two-dimensional array.")
    metadata_names = metadata["spec_name"].astype(str).tolist()
    if len(metadata_names) != len(embeddings):
        raise ValueError("Embedding metadata and NPZ have different row counts.")
    if archived_names is not None and archived_names != metadata_names:
        raise ValueError("Embedding NPZ spec_names do not match metadata order.")
    if not np.isfinite(embeddings).all():
        raise ValueError("Spectrum embeddings contain non-finite values.")
    return SpectrumEmbeddingTable(
        by_spec_name={
            spec_name: embeddings[index]
            for index, spec_name in enumerate(metadata_names)
        },
        dimension=int(embeddings.shape[1]),
    )


class RankLoopCorpusDataset(Dataset):
    """One spectrum and its complete candidate list per dataset item."""

    def __init__(
        self,
        corpus_csv: str | Path,
        embedding_table: SpectrumEmbeddingTable,
        *,
        partition: str,
        fingerprint_bits: int = 2048,
        fingerprint_radius: int = 2,
    ) -> None:
        frame = pd.read_csv(corpus_csv).fillna("")
        if missing := sorted(REQUIRED_CORPUS_COLUMNS.difference(frame.columns)):
            raise ValueError(f"RankLoop corpus is missing columns: {missing}")
        frame = frame.loc[frame["rankloop_split"] == partition].copy()
        if frame.empty:
            raise ValueError(
                f"RankLoop corpus has no rows for partition {partition!r}."
            )
        frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
        if not set(frame["label"]).issubset({0, 1}):
            raise ValueError("RankLoop labels must be binary.")

        self.embedding_table = embedding_table
        self.partition = partition
        self.fingerprint_bits = fingerprint_bits
        self.fingerprint_radius = fingerprint_radius
        self.query_names: list[str] = []
        self.rows_by_query: list[pd.DataFrame] = []
        fingerprint_cache: dict[str, np.ndarray] = {}
        self.fingerprints_by_smiles = fingerprint_cache

        for query_name, query_rows in frame.groupby("query_spec_name", sort=True):
            query_name = str(query_name)
            if query_name not in embedding_table.by_spec_name:
                raise ValueError(
                    f"Missing spectrum embedding for query {query_name!r}."
                )
            if int(query_rows["label"].sum()) != 1:
                raise ValueError(
                    f"Query {query_name!r} must contain exactly one positive."
                )
            query_connectivity = set(
                query_rows["query_inchi_key_first_block"].astype(str)
            )
            if len(query_connectivity) != 1:
                raise ValueError(
                    f"Query {query_name!r} has inconsistent connectivity labels."
                )
            for smiles in query_rows["candidate_smiles"].astype(str):
                if smiles in fingerprint_cache:
                    continue
                molecule = molecule_record_from_smiles(
                    smiles,
                    fingerprint_bits=fingerprint_bits,
                    fingerprint_radius=fingerprint_radius,
                )
                if molecule is None:
                    raise ValueError(f"Invalid candidate SMILES in corpus: {smiles!r}")
                fingerprint_cache[smiles] = molecule.fingerprint.astype(np.float32)
            self.query_names.append(query_name)
            self.rows_by_query.append(query_rows.reset_index(drop=True))

    def __len__(self) -> int:
        return len(self.query_names)

    def __getitem__(self, index: int) -> dict[str, object]:
        query_name = self.query_names[index]
        rows = self.rows_by_query[index]
        return {
            "query_spec_name": query_name,
            "query_group": str(rows["query_inchi_key_first_block"].iloc[0]),
            "spectrum_embedding": self.embedding_table.by_spec_name[query_name],
            "candidate_fingerprints": np.stack(
                [
                    self.fingerprints_by_smiles[str(smiles)]
                    for smiles in rows["candidate_smiles"]
                ]
            ),
            "candidate_labels": rows["label"].to_numpy(dtype=np.bool_),
            "candidate_smiles": rows["candidate_smiles"].astype(str).tolist(),
            "candidate_inchi_key_first_block": rows["candidate_inchi_key_first_block"]
            .astype(str)
            .tolist(),
        }


def collate_rankloop_lists(items: Sequence[dict[str, object]]) -> dict[str, object]:
    if not items:
        raise ValueError("Cannot collate an empty RankLoop batch.")
    batch_size = len(items)
    max_candidates = max(len(item["candidate_labels"]) for item in items)
    fingerprint_bits = int(items[0]["candidate_fingerprints"].shape[1])
    spectrum_dimension = int(items[0]["spectrum_embedding"].shape[0])
    spectra = np.zeros((batch_size, spectrum_dimension), dtype=np.float32)
    candidates = np.zeros(
        (batch_size, max_candidates, fingerprint_bits),
        dtype=np.float32,
    )
    candidate_mask = np.zeros((batch_size, max_candidates), dtype=np.bool_)
    positive_mask = np.zeros((batch_size, max_candidates), dtype=np.bool_)
    group_to_id: dict[str, int] = {}
    group_ids = np.zeros((batch_size,), dtype=np.int64)
    query_names: list[str] = []
    candidate_smiles: list[list[str]] = []
    candidate_connectivity: list[list[str]] = []

    for row_index, item in enumerate(items):
        item_candidates = np.asarray(item["candidate_fingerprints"], dtype=np.float32)
        item_labels = np.asarray(item["candidate_labels"], dtype=np.bool_)
        candidate_count = len(item_labels)
        if item_candidates.shape != (candidate_count, fingerprint_bits):
            raise ValueError("Candidate fingerprint dimensions are inconsistent.")
        if int(item_labels.sum()) != 1:
            raise ValueError("Each RankLoop list must contain exactly one positive.")
        spectra[row_index] = np.asarray(item["spectrum_embedding"], dtype=np.float32)
        candidates[row_index, :candidate_count] = item_candidates
        candidate_mask[row_index, :candidate_count] = True
        positive_mask[row_index, :candidate_count] = item_labels
        group = str(item["query_group"])
        group_ids[row_index] = group_to_id.setdefault(group, len(group_to_id))
        query_names.append(str(item["query_spec_name"]))
        candidate_smiles.append(list(item["candidate_smiles"]))
        candidate_connectivity.append(list(item["candidate_inchi_key_first_block"]))

    return {
        "query_spec_names": query_names,
        "query_group_ids": torch.from_numpy(group_ids),
        "spectrum_embeddings": torch.from_numpy(spectra),
        "candidate_fingerprints": torch.from_numpy(candidates),
        "candidate_mask": torch.from_numpy(candidate_mask),
        "positive_mask": torch.from_numpy(positive_mask),
        "candidate_smiles": candidate_smiles,
        "candidate_inchi_key_first_block": candidate_connectivity,
    }


class ProjectionHead(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        *,
        hidden_dimension: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dimension = hidden_dimension or max(output_dimension, input_dimension)
        self.network = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, output_dimension),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(values), dim=-1)


class MorganRankLoopDualEncoder(nn.Module):
    """Dual encoder used for data-path smoke and Morgan baseline experiments."""

    def __init__(
        self,
        spectrum_dimension: int,
        fingerprint_bits: int,
        *,
        embedding_dimension: int = 256,
        hidden_dimension: int = 512,
        dropout: float = 0.1,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.spectrum_projection = ProjectionHead(
            spectrum_dimension,
            embedding_dimension,
            hidden_dimension=hidden_dimension,
            dropout=dropout,
        )
        self.molecule_projection = ProjectionHead(
            fingerprint_bits,
            embedding_dimension,
            hidden_dimension=hidden_dimension,
            dropout=dropout,
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))

    def encode_spectra(self, spectrum_embeddings: torch.Tensor) -> torch.Tensor:
        return self.spectrum_projection(spectrum_embeddings)

    def encode_molecules(self, fingerprints: torch.Tensor) -> torch.Tensor:
        return self.molecule_projection(fingerprints)

    def forward(
        self,
        spectrum_embeddings: torch.Tensor,
        candidate_fingerprints: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spectrum_latent = self.encode_spectra(spectrum_embeddings)
        candidate_latent = self.encode_molecules(candidate_fingerprints)
        scale = self.logit_scale.clamp(max=math.log(100.0)).exp()
        logits = torch.einsum("bd,bkd->bk", spectrum_latent, candidate_latent) * scale
        return logits, spectrum_latent, candidate_latent


def positive_set_nll(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    positive_mask = positive_mask.bool()
    valid_mask = (
        torch.ones_like(positive_mask) if valid_mask is None else valid_mask.bool()
    )
    if logits.shape != positive_mask.shape or logits.shape != valid_mask.shape:
        raise ValueError("Logits and masks must have identical shapes.")
    if torch.any(positive_mask & ~valid_mask):
        raise ValueError("Positive entries must also be valid entries.")
    if not torch.all(positive_mask.any(dim=-1)):
        raise ValueError("Every row must contain at least one positive entry.")
    valid_logits = logits.masked_fill(~valid_mask, -torch.inf)
    positive_logits = logits.masked_fill(~positive_mask, -torch.inf)
    return (
        torch.logsumexp(valid_logits, dim=-1) - torch.logsumexp(positive_logits, dim=-1)
    ).mean()


def compute_rankloop_loss(
    logits: torch.Tensor,
    spectrum_latent: torch.Tensor,
    candidate_latent: torch.Tensor,
    *,
    candidate_mask: torch.Tensor,
    positive_mask: torch.Tensor,
    query_group_ids: torch.Tensor,
    logit_scale: torch.Tensor,
    symmetric_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    list_loss = positive_set_nll(logits, positive_mask, candidate_mask)
    positive_indices = positive_mask.float().argmax(dim=-1)
    batch_indices = torch.arange(len(positive_indices), device=positive_indices.device)
    positive_latent = candidate_latent[batch_indices, positive_indices]
    cross_logits = spectrum_latent @ positive_latent.transpose(0, 1) * logit_scale
    same_group = query_group_ids[:, None].eq(query_group_ids[None, :])
    spectrum_to_molecule = positive_set_nll(cross_logits, same_group)
    molecule_to_spectrum = positive_set_nll(cross_logits.transpose(0, 1), same_group)
    symmetric_loss = 0.5 * (spectrum_to_molecule + molecule_to_spectrum)
    total_loss = list_loss + symmetric_weight * symmetric_loss
    predictions = logits.masked_fill(~candidate_mask, -torch.inf).argmax(dim=-1)
    top1_correct = positive_mask[batch_indices, predictions].float().mean()
    return {
        "loss": total_loss,
        "list_loss": list_loss,
        "symmetric_loss": symmetric_loss,
        "top1_accuracy": top1_correct,
    }
