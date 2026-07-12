"""Target-blind inference utilities for RankLoop candidate reranking."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from frigid.rankloop_model import (
    DenseRankLoopDualEncoder,
    MoleculeEmbeddingTable,
    SpectrumEmbeddingTable,
)


QUERY_COLUMN_OPTIONS = ("query_spec_name", "spec_name")
CANDIDATE_COLUMN_OPTIONS = ("candidate_smiles", "smiles")


def forbidden_inference_columns(columns: list[str]) -> list[str]:
    forbidden = []
    exact = {
        "label",
        "is_positive",
        "is_target",
        "positive",
        "query_smiles",
        "query_inchi",
        "query_inchikey",
        "query_inchi_key",
        "query_inchi_key_first_block",
        "query_label_inchi_key_first_block",
    }
    for column in columns:
        normalized = str(column).strip().lower()
        if (
            normalized in exact
            or normalized.startswith("target_")
            or normalized.startswith("ground_truth")
            or "_to_target" in normalized
            or normalized.startswith("exact_match")
        ):
            forbidden.append(str(column))
    return sorted(forbidden)


def _resolve_column(columns: list[str], options: tuple[str, ...], purpose: str) -> str:
    matches = [column for column in options if column in columns]
    if len(matches) != 1:
        raise ValueError(
            f"Candidate table requires exactly one {purpose} column from {options}; "
            f"found {matches}."
        )
    return matches[0]


def load_inference_candidate_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    if frame.empty:
        raise ValueError("Candidate table is empty.")
    if forbidden := forbidden_inference_columns(list(frame.columns)):
        raise ValueError(
            "Candidate table contains forbidden target-derived columns: "
            + ", ".join(forbidden)
        )
    query_column = _resolve_column(
        list(frame.columns), QUERY_COLUMN_OPTIONS, "query identifier"
    )
    candidate_column = _resolve_column(
        list(frame.columns), CANDIDATE_COLUMN_OPTIONS, "candidate SMILES"
    )
    frame = frame.rename(
        columns={
            query_column: "query_spec_name",
            candidate_column: "candidate_smiles",
        }
    )
    for column in ("query_spec_name", "candidate_smiles"):
        frame[column] = frame[column].astype(str).str.strip()
        if (frame[column] == "").any():
            raise ValueError(f"Candidate table contains an empty {column} value.")
    duplicate = frame.duplicated(
        subset=["query_spec_name", "candidate_smiles"], keep=False
    )
    if duplicate.any():
        row = frame.loc[duplicate, ["query_spec_name", "candidate_smiles"]].iloc[0]
        raise ValueError(
            "Candidate table contains a duplicate query-candidate identity: "
            f"{row['query_spec_name']} / {row['candidate_smiles']}"
        )
    frame["rankloop_input_order"] = np.arange(len(frame), dtype=np.int64)
    return frame


def candidate_identity_sha256(frame: pd.DataFrame) -> str:
    required = {"query_spec_name", "candidate_smiles"}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"Candidate identity frame is missing columns: {missing}")
    identities = frame[["query_spec_name", "candidate_smiles"]].sort_values(
        ["query_spec_name", "candidate_smiles"], kind="mergesort"
    )
    payload = identities.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_rankloop_checkpoint(
    path: str | Path,
) -> tuple[DenseRankLoopDualEncoder, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("RankLoop checkpoint payload must be a dictionary.")
    if payload.get("architecture") != "dense_projection_dual_encoder":
        raise ValueError(
            "Unsupported RankLoop checkpoint architecture: "
            f"{payload.get('architecture')!r}"
        )
    model_config = payload.get("model_config")
    state_dict = payload.get("model_state_dict")
    if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
        raise ValueError("RankLoop checkpoint lacks model_config or model_state_dict.")
    model = DenseRankLoopDualEncoder(**model_config)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def _encode_spectrum_table(
    model: DenseRankLoopDualEncoder,
    query_names: list[str],
    table: SpectrumEmbeddingTable,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    missing = sorted(set(query_names).difference(table.by_spec_name))
    if missing:
        raise ValueError(f"Missing spectrum embedding for query {missing[0]!r}.")
    encoded: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for start in range(0, len(query_names), batch_size):
            names = query_names[start : start + batch_size]
            values = np.stack([table.by_spec_name[name] for name in names])
            latent = model.encode_spectra(
                torch.from_numpy(values).to(device, non_blocking=True)
            )
            for name, row in zip(names, latent.cpu().numpy()):
                encoded[name] = row
    return encoded


def score_candidate_frame(
    frame: pd.DataFrame,
    spectrum_table: SpectrumEmbeddingTable,
    molecule_table: MoleculeEmbeddingTable,
    model: DenseRankLoopDualEncoder,
    *,
    device: torch.device,
    batch_size: int = 4096,
) -> pd.DataFrame:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    required = {"query_spec_name", "candidate_smiles", "rankloop_input_order"}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"Candidate frame is missing columns: {missing}")
    query_names = sorted(set(frame["query_spec_name"].astype(str)))
    missing_molecules = sorted(
        set(frame["candidate_smiles"].astype(str)).difference(molecule_table.by_smiles)
    )
    if missing_molecules:
        raise ValueError(
            f"Missing molecule embedding for candidate {missing_molecules[0]!r}."
        )

    model = model.to(device)
    spectrum_latent = _encode_spectrum_table(
        model,
        query_names,
        spectrum_table,
        device=device,
        batch_size=batch_size,
    )
    scale = float(model.logit_scale.clamp(max=math.log(100.0)).exp().detach().cpu())
    scores = np.empty(len(frame), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(frame), batch_size):
            stop = min(start + batch_size, len(frame))
            batch = frame.iloc[start:stop]
            molecule_values = np.stack(
                [molecule_table.by_smiles[value] for value in batch["candidate_smiles"]]
            )
            molecule_latent = model.encode_molecules(
                torch.from_numpy(molecule_values).to(device, non_blocking=True)
            )
            query_values = np.stack(
                [spectrum_latent[value] for value in batch["query_spec_name"]]
            )
            query_latent = torch.from_numpy(query_values).to(device, non_blocking=True)
            scores[start:stop] = (
                ((query_latent * molecule_latent).sum(dim=-1) * scale).cpu().numpy()
            )
    if not np.isfinite(scores).all():
        raise AssertionError("RankLoop produced non-finite candidate scores.")

    ranked = frame.copy()
    ranked["rankloop_score"] = scores
    ranked = ranked.sort_values(
        ["query_spec_name", "rankloop_score", "rankloop_input_order"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["rankloop_rank"] = (
        ranked.groupby("query_spec_name", sort=False).cumcount() + 1
    )
    return ranked
