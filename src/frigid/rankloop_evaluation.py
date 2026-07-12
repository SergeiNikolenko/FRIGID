"""Post-hoc evaluation of target-blind RankLoop candidate rankings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from frigid.rankloop_corpus import molecule_record_from_smiles
from frigid.rankloop_inference import (
    candidate_identity_sha256,
    forbidden_inference_columns,
)


def _resolve_column(columns: list[str], options: tuple[str, ...], purpose: str) -> str:
    matches = [column for column in options if column in columns]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {purpose} column from {options}; found {matches}."
        )
    return matches[0]


def load_ranked_candidate_frame(
    path: str | Path,
    *,
    rank_column: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path).fillna("")
    if frame.empty:
        raise ValueError("Ranked candidate table is empty.")
    if forbidden := forbidden_inference_columns(list(frame.columns)):
        raise ValueError(
            "Ranked candidate table contains target-derived columns: "
            + ", ".join(forbidden)
        )
    if rank_column not in frame.columns:
        raise ValueError(f"Ranked candidate table lacks {rank_column!r}.")
    query_column = _resolve_column(
        list(frame.columns), ("query_spec_name", "spec_name"), "query identifier"
    )
    smiles_column = _resolve_column(
        list(frame.columns), ("candidate_smiles", "smiles"), "candidate SMILES"
    )
    frame = frame.rename(
        columns={query_column: "query_spec_name", smiles_column: "candidate_smiles"}
    )
    frame[rank_column] = pd.to_numeric(frame[rank_column], errors="raise").astype(int)
    if (frame[rank_column] <= 0).any():
        raise ValueError("Candidate ranks must be positive integers.")
    for column in ("query_spec_name", "candidate_smiles"):
        frame[column] = frame[column].astype(str).str.strip()
        if (frame[column] == "").any():
            raise ValueError(f"Ranked candidate table contains empty {column} values.")
    if frame.duplicated(["query_spec_name", "candidate_smiles"]).any():
        raise ValueError(
            "Ranked candidate table contains duplicate candidate identities."
        )
    if frame.duplicated(["query_spec_name", rank_column]).any():
        raise ValueError(
            "Ranked candidate table contains duplicate ranks within a query."
        )
    for query_name, rows in frame.groupby("query_spec_name", sort=False):
        ranks = sorted(rows[rank_column].tolist())
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f"Candidate ranks are not contiguous for {query_name!r}.")
    return frame


def load_target_metadata(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path).fillna("")
    query_column = _resolve_column(
        list(frame.columns), ("spec_name", "query_spec_name"), "query identifier"
    )
    smiles_column = _resolve_column(
        list(frame.columns), ("target_smiles", "smiles"), "target SMILES"
    )
    inchi_column = _resolve_column(
        list(frame.columns),
        ("target_inchi_key", "inchi_key", "inchikey"),
        "target InChIKey",
    )
    targets = frame[[query_column, smiles_column, inchi_column]].rename(
        columns={
            query_column: "spec_name",
            smiles_column: "target_smiles",
            inchi_column: "target_inchi_key",
        }
    )
    for column in targets.columns:
        targets[column] = targets[column].astype(str).str.strip()
        if (targets[column] == "").any():
            raise ValueError(f"Target metadata contains empty {column} values.")
    if targets["spec_name"].duplicated().any():
        raise ValueError("Target metadata contains duplicate spec_name values.")
    targets["target_inchi_key"] = targets["target_inchi_key"].str.split("-", n=1).str[0]
    return targets


def validate_identical_candidate_pool(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> str:
    reference_hash = candidate_identity_sha256(reference)
    candidate_hash = candidate_identity_sha256(candidate)
    if reference_hash != candidate_hash:
        raise ValueError(
            "RankLoop candidate pool differs from the frozen reference pool: "
            f"{reference_hash} != {candidate_hash}"
        )
    return reference_hash


def _tanimoto(left: np.ndarray, right: np.ndarray) -> float:
    left_bool = np.asarray(left, dtype=bool)
    right_bool = np.asarray(right, dtype=bool)
    union = np.logical_or(left_bool, right_bool).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left_bool, right_bool).sum() / union)


def evaluate_ranked_candidates(
    ranked: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    rank_column: str,
    method_name: str,
    top_k: int = 10,
    fingerprint_bits: int = 4096,
    fingerprint_radius: int = 2,
) -> pd.DataFrame:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    target_by_query = targets.set_index("spec_name")
    query_names = sorted(set(ranked["query_spec_name"].astype(str)))
    missing_targets = sorted(set(query_names).difference(target_by_query.index))
    if missing_targets:
        raise ValueError(f"Missing target metadata for {missing_targets[0]!r}.")

    molecule_cache = {}

    def molecule(smiles: str):
        if smiles not in molecule_cache:
            record = molecule_record_from_smiles(
                smiles,
                fingerprint_bits=fingerprint_bits,
                fingerprint_radius=fingerprint_radius,
            )
            if record is None:
                raise ValueError(f"Could not featurize candidate SMILES {smiles!r}.")
            molecule_cache[smiles] = record
        return molecule_cache[smiles]

    detailed_rows = []
    for query_name, query_rows in ranked.groupby("query_spec_name", sort=True):
        query_rows = query_rows.sort_values(rank_column, kind="mergesort")
        target = target_by_query.loc[str(query_name)]
        target_record = molecule(str(target["target_smiles"]))
        target_inchi = str(target["target_inchi_key"])
        if target_record.inchi_key_first_block != target_inchi:
            target_inchi = target_record.inchi_key_first_block

        similarities = []
        exact = []
        for smiles in query_rows["candidate_smiles"].astype(str):
            candidate = molecule(smiles)
            similarities.append(
                _tanimoto(target_record.fingerprint, candidate.fingerprint)
            )
            exact.append(candidate.inchi_key_first_block == target_inchi)
        similarities_array = np.asarray(similarities, dtype=np.float64)
        exact_array = np.asarray(exact, dtype=bool)
        top_count = min(top_k, len(query_rows))
        target_positions = np.flatnonzero(exact_array)
        target_rank = int(target_positions[0] + 1) if len(target_positions) else 0
        oracle_tanimoto = float(similarities_array.max())
        tanimoto_top1 = float(similarities_array[0])
        detailed_rows.append(
            {
                "spec_name": str(query_name),
                "target_smiles": str(target["target_smiles"]),
                "target_inchi_key": target_inchi,
                "fingerprint_source": "rankloop_posthoc",
                "candidate_method": method_name,
                "exact_match_top1": float(exact_array[0]),
                "exact_match_top10": float(exact_array[:top_count].any()),
                "tanimoto_top1": tanimoto_top1,
                "tanimoto_top10": float(similarities_array[:top_count].max()),
                "candidate_recall_exact": float(exact_array.any()),
                "candidate_oracle_tanimoto": oracle_tanimoto,
                "ranking_regret_top1": oracle_tanimoto - tanimoto_top1,
                "target_rank": target_rank,
                "total_valid": int(len(query_rows)),
            }
        )
    return pd.DataFrame(detailed_rows)
