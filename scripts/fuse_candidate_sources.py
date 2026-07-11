#!/usr/bin/env python
"""Fuse multiple candidate sources for reproducible benchmark evaluation."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from retrieve_train_candidates import (
    _sha256_hex,
    compute_morgan_fingerprint,
    compute_tanimoto_similarity,
)


def _resolve_column(
    frame: pd.DataFrame,
    names: tuple[str, ...],
    purpose: str,
) -> str:
    for name in names:
        if name in frame.columns:
            return name
    available = ", ".join(frame.columns)
    raise ValueError(f"Missing {purpose} column. Available: {available}")


def _resolve_optional_column(
    frame: pd.DataFrame,
    names: tuple[str, ...],
) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _parse_source_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Source argument must be in NAME=PATH form: {value!r}"
        )
    source_name, source_path = value.split("=", maxsplit=1)
    source_name = source_name.strip()
    if not source_name:
        raise argparse.ArgumentTypeError(f"Source name cannot be empty: {value!r}")
    path = Path(source_path.strip()).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Source file does not exist: {path}")
    return source_name, path


def _normalize_inchi_key(value: str) -> str:
    return str(value).strip().split("-", maxsplit=1)[0]


def _compute_inchi_key_first_block(smiles: str) -> str:
    try:
        from rdkit import Chem
    except Exception as exc:
        raise RuntimeError("RDKit is required to compute InChIKeys.") from exc

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    inchi = Chem.MolToInchiKey(mol)
    return inchi.split("-")[0]


def _parse_rank(value: Any, source: str, row_index: int, column: str) -> int:
    raw = str(value).strip()
    if not raw:
        raise ValueError(f"{source} rank column {column!r} at row {row_index} is empty.")
    try:
        rank = int(float(raw))
    except ValueError as exc:
        raise ValueError(
            f"{source} rank value {value!r} at row {row_index} is not a number."
        ) from exc
    if rank < 0:
        raise ValueError(
            f"{source} rank value must be non-negative; got {rank} at row {row_index}."
        )
    return rank


def infer_source_schema(frame: pd.DataFrame) -> tuple[str, str, str]:
    query_col = _resolve_column(
        frame,
        ("query_spec_name", "spec_name"),
        "query",
    )
    rank_col = _resolve_column(
        frame,
        ("rank",),
        "rank",
    )
    smiles_col = _resolve_column(
        frame,
        ("candidate_smiles", "smiles"),
        "smiles",
    )
    return query_col, rank_col, smiles_col


@dataclass(frozen=True)
class SourceCandidate:
    query_spec_name: str
    source_name: str
    source_rank: int
    smiles: str
    candidate_spec_name: str | None
    inchi_key_first_block: str
    fingerprint: np.ndarray


@dataclass(frozen=True)
class RankedCandidate:
    rank: int
    source_name: str
    source_rank: int
    source_candidate_rank: int
    candidate_smiles: str
    candidate_spec_name: str | None
    candidate_inchi_key_first_block: str
    tanimoto_to_mist: float
    fingerprint: np.ndarray


def _load_source_rows(
    source_name: str,
    source_path: Path,
    fingerprint_bits: int,
    fingerprint_radius: int,
) -> list[SourceCandidate]:
    frame = pd.read_csv(source_path).fillna("")
    query_col, rank_col, smiles_col = infer_source_schema(frame)
    candidate_spec_col = _resolve_optional_column(
        frame,
        ("candidate_spec_name", "candidate_name", "spec_name"),
    )

    rows: list[SourceCandidate] = []
    for index, row in frame.iterrows():
        query_spec_name = str(row[query_col]).strip()
        smiles = str(row[smiles_col]).strip()
        if not query_spec_name:
            raise ValueError(f"{source_name}: empty query spec name at row {index}.")
        if not smiles:
            raise ValueError(
                f"{source_name}: empty smiles for {query_spec_name} at row {index}."
            )

        source_rank = _parse_rank(row[rank_col], source_name, index, rank_col)
        candidate_spec_name = (
            str(row[candidate_spec_col]).strip() if candidate_spec_col else None
        )
        fingerprint = compute_morgan_fingerprint(
            smiles,
            bits=fingerprint_bits,
            radius=fingerprint_radius,
        )
        if fingerprint is None:
            raise ValueError(
                f"{source_name}: could not compute fingerprint for {query_spec_name} "
                f"smiles {smiles!r}."
            )
        inchi_key_first_block = _compute_inchi_key_first_block(smiles)
        if not inchi_key_first_block:
            raise ValueError(
                f"{source_name}: could not compute InChIKey first block for "
                f"{query_spec_name} candidate {smiles!r}."
            )
        rows.append(
            SourceCandidate(
                query_spec_name=query_spec_name,
                source_name=source_name,
                source_rank=source_rank,
                smiles=smiles,
                candidate_spec_name=candidate_spec_name,
                inchi_key_first_block=inchi_key_first_block,
                fingerprint=fingerprint,
            )
        )
    return rows


def _dedupe_key(candidate: SourceCandidate) -> tuple[int, str, str, str]:
    return (
        candidate.source_rank,
        candidate.source_name,
        candidate.candidate_spec_name or "",
        candidate.smiles,
    )


def deduplicate_by_inchikey_first_block(
    candidates: list[SourceCandidate],
) -> list[SourceCandidate]:
    deduped: dict[str, SourceCandidate] = {}
    for candidate in candidates:
        current = deduped.get(candidate.inchi_key_first_block)
        if current is None or _dedupe_key(candidate) < _dedupe_key(current):
            deduped[candidate.inchi_key_first_block] = candidate
    return [
        entry
        for _, entry in sorted(
            (_dedupe_key(candidate), candidate) for candidate in deduped.values()
        )
    ]


def _rank_key(candidate: SourceCandidate, similarity: float) -> tuple:
    return (-similarity, candidate.source_rank, candidate.source_name, candidate.smiles)


def rank_query_candidates(
    query_fp: np.ndarray,
    candidates: list[SourceCandidate],
    top_k: int,
) -> list[RankedCandidate]:
    if top_k <= 0:
        raise ValueError("--top-k must be positive.")
    scored: list[tuple[float, SourceCandidate]] = []
    for candidate in candidates:
        similarity = compute_tanimoto_similarity(query_fp, candidate.fingerprint)
        scored.append((similarity, candidate))
    scored.sort(key=lambda item: _rank_key(item[1], item[0]))
    ranked: list[RankedCandidate] = []
    for idx, (similarity, candidate) in enumerate(scored[:top_k], start=1):
        ranked.append(
            RankedCandidate(
                rank=idx,
                source_name=candidate.source_name,
                source_rank=candidate.source_rank,
                source_candidate_rank=candidate.source_rank,
                candidate_smiles=candidate.smiles,
                candidate_spec_name=candidate.candidate_spec_name,
                candidate_inchi_key_first_block=candidate.inchi_key_first_block,
                tanimoto_to_mist=similarity,
                fingerprint=candidate.fingerprint,
            )
        )
    return ranked


def evaluate_ranked_predictions(
    target_inchi_key_first_block: str,
    target_fingerprint: np.ndarray | None,
    ranked: list[RankedCandidate],
    top_k: int,
) -> dict[str, float]:
    top_predictions = ranked[:top_k]
    top_inchi_blocks = [entry.candidate_inchi_key_first_block for entry in top_predictions]

    if not top_predictions:
        return {
            "exact_match_top1": 0.0,
            "exact_match_top10": 0.0,
            "tanimoto_top1": 0.0,
            "tanimoto_top10": 0.0,
            "total_formula_matched": 0.0,
        }

    target_tanimotos: list[float] = []
    if target_fingerprint is None:
        target_tanimotos = [0.0 for _ in top_predictions]
    else:
        target_tanimotos = [
            compute_tanimoto_similarity(target_fingerprint, candidate.fingerprint)
            for candidate in top_predictions
        ]

    exact_top1 = 1.0 if top_inchi_blocks and top_inchi_blocks[0] == target_inchi_key_first_block else 0.0
    exact_top10 = 1.0 if target_inchi_key_first_block in top_inchi_blocks[:10] else 0.0
    tanimoto_top1 = target_tanimotos[0] if target_tanimotos else 0.0
    tanimoto_top10 = max(target_tanimotos) if target_tanimotos else 0.0

    return {
        "exact_match_top1": float(exact_top1),
        "exact_match_top10": float(exact_top10),
        "tanimoto_top1": float(tanimoto_top1),
        "tanimoto_top10": float(tanimoto_top10),
        "total_formula_matched": 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse candidate source CSVs by InChIKey and rerank by MIST similarity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        type=_parse_source_argument,
        help="Repeated source spec in NAME=PATH form.",
    )
    parser.add_argument(
        "--mist-metadata-csv",
        required=True,
        help="MIST exported metadata.csv with spec_name/target smiles and inchi.",
    )
    parser.add_argument(
        "--mist-fingerprints-npz",
        required=True,
        help="Exported fingerprints.npz containing mist_binary.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for output files.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of predictions to output.")
    parser.add_argument("--fingerprint-bits", type=int, default=4096)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    return parser.parse_args()


def _load_metadata(
    metadata_path: Path,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    frame = pd.read_csv(metadata_path).fillna("")
    spec_col = _resolve_column(frame, ("spec_name",), "spec_name")
    smiles_col = _resolve_column(frame, ("target_smiles", "smiles"), "target smiles")
    inchi_col = _resolve_column(
        frame,
        ("target_inchi_key", "inchi_key_first_block", "inchi_key"),
        "target inchi key",
    )

    rows: list[str] = []
    by_spec: dict[str, dict[str, str]] = {}
    for _, row in frame.iterrows():
        spec_name = str(row[spec_col]).strip()
        if not spec_name:
            raise ValueError("metadata.csv contains empty spec_name.")
        if spec_name in by_spec:
            raise ValueError(f"Duplicate spec_name in metadata.csv: {spec_name}")
        by_spec[spec_name] = {
            "target_smiles": str(row[smiles_col]).strip(),
            "target_inchi_key": _normalize_inchi_key(str(row[inchi_col]).strip()),
        }
        rows.append(spec_name)
    return rows, by_spec


def run_fuse_candidate_sources(
    source_specs: list[tuple[str, Path]],
    mist_metadata_csv: Path,
    mist_fingerprints_npz: Path,
    output_dir: Path,
    top_k: int,
    fingerprint_bits: int,
    fingerprint_radius: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if fingerprint_bits <= 0:
        raise ValueError("--fingerprint-bits must be positive.")
    if fingerprint_radius < 0:
        raise ValueError("--fingerprint-radius must be non-negative.")

    query_order, metadata_by_spec = _load_metadata(mist_metadata_csv)

    with np.load(mist_fingerprints_npz, mmap_mode="r") as loaded:
        if "mist_binary" not in loaded.files:
            raise ValueError("fingerprints.npz must contain mist_binary.")
        mist_binary = np.asarray(loaded["mist_binary"], dtype=np.float32)

    if mist_binary.ndim != 2:
        raise ValueError("mist_binary must be a 2D array.")
    if mist_binary.shape[1] != fingerprint_bits:
        raise ValueError(
            "mist_binary width does not match --fingerprint-bits: "
            f"{mist_binary.shape[1]} != {fingerprint_bits}"
        )

    source_rows_by_query: dict[str, list[SourceCandidate]] = defaultdict(list)
    source_rows: dict[str, int] = {}
    for source_name, source_path in source_specs:
        rows = _load_source_rows(
            source_name=source_name,
            source_path=source_path,
            fingerprint_bits=fingerprint_bits,
            fingerprint_radius=fingerprint_radius,
        )
        source_rows[source_name] = len(rows)
        for row in rows:
            source_rows_by_query[row.query_spec_name].append(row)

    source_spec_names = [name for name, _ in source_specs]
    detailed_rows: list[dict[str, Any]] = []
    predictions_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    metrics_top1: list[float] = []
    metrics_top10: list[float] = []
    candidate_metrics_top1: list[float] = []
    candidate_metrics_top10: list[float] = []

    for query_index, query_spec_name in enumerate(query_order):
        metadata = metadata_by_spec[query_spec_name]
        target_smiles = metadata["target_smiles"]
        target_inchi_block = metadata["target_inchi_key"]

        if not target_smiles:
            raise ValueError(f"Missing target smiles for {query_spec_name}")
        if not target_inchi_block:
            raise ValueError(f"Missing target inchi_key for {query_spec_name}")
        if query_index >= mist_binary.shape[0]:
            raise ValueError(
                f"metadata and mist_binary are misaligned: missing row for {query_spec_name}"
            )
        query_fp = mist_binary[query_index]

        raw_candidates = source_rows_by_query.get(query_spec_name, [])
        unique_candidates = deduplicate_by_inchikey_first_block(raw_candidates)
        ranked = rank_query_candidates(query_fp, unique_candidates, top_k=top_k)
        target_fingerprint = compute_morgan_fingerprint(
            target_smiles,
            bits=fingerprint_bits,
            radius=fingerprint_radius,
        )
        metrics = evaluate_ranked_predictions(
            target_inchi_key_first_block=target_inchi_block,
            target_fingerprint=target_fingerprint,
            ranked=ranked,
            top_k=top_k,
        )

        mist_tanimoto = (
            compute_tanimoto_similarity(target_fingerprint, query_fp)
            if target_fingerprint is not None
            else 0.0
        )
        total_generated = len(raw_candidates)

        detailed_rows.append(
            {
                "spec_name": query_spec_name,
                "target_smiles": target_smiles,
                "target_inchi_key": target_inchi_block,
                "mist_tanimoto": mist_tanimoto,
                "fingerprint_source": "fused_candidates",
                "exact_match_top1": metrics["exact_match_top1"],
                "exact_match_top10": metrics["exact_match_top10"],
                "tanimoto_top1": metrics["tanimoto_top1"],
                "tanimoto_top10": metrics["tanimoto_top10"],
                "total_formula_matched": metrics["total_formula_matched"],
                "total_valid": len(ranked),
                "total_generated": total_generated,
            }
        )

        prediction_row = {"true_smiles": target_smiles, "name": query_spec_name}
        for index in range(top_k):
            key = f"pred_smiles_{index + 1}"
            if index < len(ranked):
                prediction_row[key] = ranked[index].candidate_smiles
            else:
                prediction_row[key] = ""
        predictions_rows.append(prediction_row)

        for ranked_entry in ranked:
            score_rows.append(
                {
                    "query_spec_name": query_spec_name,
                    "rank": ranked_entry.rank,
                    "source_name": ranked_entry.source_name,
                    "source_rank": ranked_entry.source_rank,
                    "candidate_rank": ranked_entry.source_candidate_rank,
                    "candidate_smiles": ranked_entry.candidate_smiles,
                    "candidate_spec_name": ranked_entry.candidate_spec_name or "",
                    "candidate_inchi_key_first_block": ranked_entry.candidate_inchi_key_first_block,
                    "tanimoto_to_mist": ranked_entry.tanimoto_to_mist,
                }
            )

        metrics_top1.append(metrics["exact_match_top1"])
        metrics_top10.append(metrics["exact_match_top10"])
        candidate_metrics_top1.append(metrics["tanimoto_top1"])
        candidate_metrics_top10.append(metrics["tanimoto_top10"])

    if not detailed_rows:
        raise ValueError("No queries were processed from metadata.")

    exact_match_top1 = float(np.mean(metrics_top1)) if metrics_top1 else 0.0
    exact_match_top10 = float(np.mean(metrics_top10)) if metrics_top10 else 0.0
    tanimoto_top1 = float(np.mean(candidate_metrics_top1)) if candidate_metrics_top1 else 0.0
    tanimoto_top10 = (
        float(np.mean(candidate_metrics_top10)) if candidate_metrics_top10 else 0.0
    )

    aggregate_statistics = {
        "schema_version": 1,
        "n_queries": len(detailed_rows),
        "top_k": top_k,
        "fingerprint_bits": fingerprint_bits,
        "fingerprint_radius": fingerprint_radius,
        "source_count": len(source_specs),
        "source_names": source_spec_names,
        "source_rows": source_rows,
        "exact_match_top1": exact_match_top1,
        "exact_match_top10": exact_match_top10,
        "tanimoto_top1_mean": tanimoto_top1,
        "tanimoto_top10_mean": tanimoto_top10,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detailed_rows).to_csv(
        output_dir / "detailed_results.csv",
        index=False,
    )
    pd.DataFrame(predictions_rows).to_csv(
        output_dir / "predictions.csv",
        index=False,
    )
    pd.DataFrame(score_rows).to_csv(
        output_dir / "prediction_scores.csv",
        index=False,
    )
    with (output_dir / "aggregate_statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate_statistics, handle, indent=2)

    source_rows_payload = []
    for source_name, source_path in source_specs:
        source_rows_payload.append(
            {
                "name": source_name,
                "path": str(source_path),
                "sha256": _sha256_hex(source_path),
                "row_count": source_rows[source_name],
            }
        )
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "script": str(Path(__file__).resolve()),
                "mist_metadata_csv": {
                    "path": str(mist_metadata_csv),
                    "sha256": _sha256_hex(mist_metadata_csv),
                },
                "mist_fingerprints_npz": {
                    "path": str(mist_fingerprints_npz),
                    "sha256": _sha256_hex(mist_fingerprints_npz),
                },
                "sources": source_rows_payload,
                "parameters": {
                    "top_k": top_k,
                    "fingerprint_bits": fingerprint_bits,
                    "fingerprint_radius": fingerprint_radius,
                },
                "outputs": {
                    "prediction_scores_csv": str(output_dir / "prediction_scores.csv"),
                    "predictions_csv": str(output_dir / "predictions.csv"),
                    "detailed_results_csv": str(output_dir / "detailed_results.csv"),
                    "aggregate_statistics_json": str(
                        output_dir / "aggregate_statistics.json"
                    ),
                    "run_manifest_json": str(output_dir / "run_manifest.json"),
                },
            },
            handle,
            indent=2,
        )

    return (
        pd.DataFrame(predictions_rows),
        pd.DataFrame(score_rows),
        pd.DataFrame(detailed_rows),
    )


def main() -> int:
    args = parse_args()
    source_specs = args.source

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_names = [name for name, _ in source_specs]
    if len(set(source_names)) != len(source_names):
        duplicates = sorted(
            name for name in set(source_names) if source_names.count(name) > 1
        )
        raise ValueError(f"Duplicate source names are not allowed: {duplicates}")

    mist_metadata_csv = Path(args.mist_metadata_csv).expanduser().resolve()
    mist_fingerprints_npz = Path(args.mist_fingerprints_npz).expanduser().resolve()

    _, _, _ = run_fuse_candidate_sources(
        source_specs=source_specs,
        mist_metadata_csv=mist_metadata_csv,
        mist_fingerprints_npz=mist_fingerprints_npz,
        output_dir=output_dir,
        top_k=args.top_k,
        fingerprint_bits=args.fingerprint_bits,
        fingerprint_radius=args.fingerprint_radius,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
