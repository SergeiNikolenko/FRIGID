#!/usr/bin/env python
"""Train-only molecular retrieval baseline for MSG candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd

ELEMENT_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*)")


def _sha256_hex(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_tanimoto_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    intersection = np.sum(np.minimum(fp1, fp2))
    union = np.sum(np.maximum(fp1, fp2))
    return float(intersection / union) if union > 0 else 0.0


def normalize_formula(formula: str | None) -> str | None:
    if not formula:
        return None
    matches = ELEMENT_PATTERN.findall(str(formula))
    if not matches:
        return str(formula)

    counts: dict[str, int] = {}
    for element, count_str in matches:
        if element:
            count = int(count_str) if count_str else 1
            counts[element] = counts.get(element, 0) + count

    ordered_elements: list[str] = []
    if "C" in counts:
        ordered_elements.append("C")
        if "H" in counts:
            ordered_elements.append("H")
    ordered_elements.extend(
        sorted(element for element in counts.keys() if element not in ordered_elements)
    )
    return "".join(
        f"{element}{counts[element] if counts[element] != 1 else ''}"
        for element in ordered_elements
    )


def compute_morgan_fingerprint(
    smiles: str, bits: int, radius: int
) -> np.ndarray | None:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        from rdkit import RDLogger
    except Exception as exc:
        raise RuntimeError(
            "RDKit is required to compute Morgan fingerprints for this script."
        ) from exc

    RDLogger.DisableLog("rdApp.*")
    if not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits)
        array = np.zeros((bits,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, array)
        return array
    except Exception:
        return None


def _resolve_column(df: pd.DataFrame, names: tuple[str, ...], purpose: str) -> str:
    for name in names:
        if name in df.columns:
            return name
    available = ", ".join(df.columns)
    raise ValueError(f"{purpose} column not found. Available: {available}")


def _normalize_split_split(value: Any) -> str:
    return str(value).strip().lower()


@dataclass(frozen=True)
class MoleculeCandidate:
    spec_name: str
    smiles: str
    inchi_key_first_block: str
    formula: str
    fingerprint: np.ndarray


@dataclass(frozen=True)
class RankedCandidate:
    rank: int
    candidate: MoleculeCandidate
    tanimoto: float
    formula_match: bool


def resolve_split_assignments(
    split_rows: list[tuple[str, str]],
    query_splits: tuple[str, ...] | list[str],
) -> tuple[list[str], list[str]]:
    """Return train specs and query specs in the source-file order."""
    seen: set[str] = set()
    train_specs: list[str] = []
    query_specs: list[str] = []
    query_set = {_normalize_split_split(s) for s in query_splits}

    for index, (spec_name, split_name) in enumerate(split_rows):
        if not spec_name:
            raise ValueError(f"Split entry has empty spec name at row {index}.")
        if spec_name in seen:
            raise ValueError(f"Duplicate spec name in split file: {spec_name}")
        seen.add(spec_name)

        normalized_split = _normalize_split_split(split_name)
        if normalized_split == "train":
            train_specs.append(spec_name)
        if normalized_split in query_set:
            query_specs.append(spec_name)
    return train_specs, query_specs


def deduplicate_by_inchikey_first_block(
    candidates: list[MoleculeCandidate],
) -> list[MoleculeCandidate]:
    seen: set[str] = set()
    deduplicated: list[MoleculeCandidate] = []
    for candidate in candidates:
        key = candidate.inchi_key_first_block
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(candidate)
    return deduplicated


def rank_train_candidates(
    query_fp: np.ndarray,
    query_formula: str,
    train_library: list[MoleculeCandidate],
    top_k: int,
) -> list[RankedCandidate]:
    """Rank candidates by formula match first, then Tanimoto similarity."""
    scored: list[tuple[int, float, int, MoleculeCandidate]] = []
    for idx, candidate in enumerate(train_library):
        tanimoto = compute_tanimoto_similarity(query_fp, candidate.fingerprint)
        formula_match = 1 if candidate.formula and candidate.formula == query_formula else 0
        scored.append((formula_match, tanimoto, idx, candidate))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    ranked: list[RankedCandidate] = []
    for rank, (formula_match, tanimoto, _, candidate) in enumerate(scored[:top_k], start=1):
        ranked.append(
            RankedCandidate(
                rank=rank,
                candidate=candidate,
                tanimoto=tanimoto,
                formula_match=bool(formula_match),
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
    top_inchis = [entry.candidate.inchi_key_first_block for entry in top_predictions]

    if not top_predictions:
        return {
            "exact_match_top1": 0.0,
            "exact_match_top10": 0.0,
            "tanimoto_top1": 0.0,
            "tanimoto_top10": 0.0,
            "formula_match_count": 0.0,
        }

    formula_tanimotos: list[float] = []
    if target_fingerprint is None:
        formula_tanimotos = [0.0 for _ in top_predictions]
    else:
        formula_tanimotos = [
            compute_tanimoto_similarity(target_fingerprint, candidate.candidate.fingerprint)
            for candidate in top_predictions
        ]

    exact_top1 = 1.0 if top_inchis and top_inchis[0] == target_inchi_key_first_block else 0.0
    exact_top10 = 1.0 if target_inchi_key_first_block in top_inchis[:10] else 0.0
    tanimoto_top1 = formula_tanimotos[0] if formula_tanimotos else 0.0
    tanimoto_top10 = float(np.max(formula_tanimotos)) if formula_tanimotos else 0.0

    return {
        "exact_match_top1": float(exact_top1),
        "exact_match_top10": float(exact_top10),
        "tanimoto_top1": float(tanimoto_top1),
        "tanimoto_top10": float(tanimoto_top10),
        "formula_match_count": float(
            sum(1 for candidate in top_predictions if candidate.formula_match)
        ),
    }


def validate_query_leakage(
    query_inchi_key_first_blocks: list[str],
    train_inchi_keys: set[str],
    allow_query_in_train: bool,
) -> None:
    if allow_query_in_train:
        return
    overlap = sorted(
        key
        for key in query_inchi_key_first_blocks
        if key and key in train_inchi_keys
    )
    if overlap:
        raise ValueError(
            "Query first-block InChI keys overlap the train split: "
            + ", ".join(overlap[:20])
        )


def select_exported_query_specs(
    metadata_spec_names: list[str], allowed_query_specs: set[str]
) -> list[str]:
    unexpected_specs = [
        spec_name
        for spec_name in metadata_spec_names
        if spec_name not in allowed_query_specs
    ]
    if unexpected_specs:
        raise ValueError(
            "Exported MIST metadata contains spectra outside --query-splits: "
            + ", ".join(unexpected_specs[:20])
        )
    return metadata_spec_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible train-only molecule retrieval baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--labels-tsv", required=True, help="MSG labels TSV file.")
    parser.add_argument("--split-tsv", required=True, help="MSG split TSV file.")
    parser.add_argument(
        "--mist-metadata-csv",
        required=True,
        help="Exported MIST metadata.csv path.",
    )
    parser.add_argument(
        "--mist-fingerprints-npz",
        required=True,
        help="Exported fingerprints.npz containing mist_binary.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for output files.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of candidates to return.")
    parser.add_argument("--fingerprint-bits", type=int, default=4096)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument(
        "--query-splits",
        nargs="+",
        default=("val", "test"),
        help="Split names to run retrieval for (default: val test).",
    )
    parser.add_argument(
        "--allow-query-in-train",
        action="store_true",
        help=(
            "Allow query molecules whose InChIKey first block is present in train "
            "split (off by default)."
        ),
    )
    return parser.parse_args()


def _build_library_from_train(
    labels_by_spec: dict[str, dict[str, str]],
    train_specs: list[str],
    bits: int,
    radius: int,
    smiles_col: str,
    inchi_col: str,
    formula_col: str,
) -> list[MoleculeCandidate]:
    candidates: list[MoleculeCandidate] = []
    seen_inchi_keys: set[str] = set()
    for spec_name in train_specs:
        if spec_name not in labels_by_spec:
            raise ValueError(f"Train spec missing from labels.tsv: {spec_name}")
        row = labels_by_spec[spec_name]
        smiles = row.get(smiles_col, "").strip()
        if not smiles:
            raise ValueError(f"Missing smiles for train spec: {spec_name}")

        inchi_key = row.get(inchi_col, "").strip()
        if not inchi_key:
            raise ValueError(f"Missing InChIKey for train spec: {spec_name}")
        inchi_key_first_block = inchi_key.split("-")[0]
        if inchi_key_first_block in seen_inchi_keys:
            continue
        seen_inchi_keys.add(inchi_key_first_block)

        formula_raw = row.get(formula_col, "")
        formula = normalize_formula(formula_raw) if formula_raw else ""

        fingerprint = compute_morgan_fingerprint(smiles, bits, radius)
        if fingerprint is None:
            raise ValueError(f"Could not compute train fingerprint for {spec_name}")
        candidates.append(
            MoleculeCandidate(
                spec_name=spec_name,
                smiles=smiles,
                inchi_key_first_block=inchi_key_first_block,
                formula=formula,
                fingerprint=fingerprint,
            )
        )

    deduplicated = deduplicate_by_inchikey_first_block(candidates)
    if not deduplicated:
        raise ValueError("No valid train candidates found.")
    return deduplicated


def main() -> int:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")
    if args.fingerprint_bits <= 0:
        raise ValueError("--fingerprint-bits must be positive.")
    if args.fingerprint_radius < 0:
        raise ValueError("--fingerprint-radius must be non-negative.")

    labels_path = Path(args.labels_tsv).expanduser().resolve()
    split_path = Path(args.split_tsv).expanduser().resolve()
    metadata_path = Path(args.mist_metadata_csv).expanduser().resolve()
    fp_path = Path(args.mist_fingerprints_npz).expanduser().resolve()
    output_path = Path(args.output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(labels_path, sep="\t", dtype=str).fillna("")
    split_df = pd.read_csv(split_path, sep="\t", dtype=str).fillna("")

    labels_spec_col = _resolve_column(labels_df, ("spec", "spec_name"), "labels")
    labels_smiles_col = _resolve_column(labels_df, ("smiles",), "labels")
    labels_formula_col = _resolve_column(labels_df, ("formula",), "labels")
    labels_inchi_col = _resolve_column(labels_df, ("inchi_key", "inchikey"), "labels")

    split_spec_col = _resolve_column(split_df, ("spec", "name", "spec_name"), "split")
    split_name_col = _resolve_column(split_df, ("split",), "split")

    labels_by_spec: dict[str, dict[str, str]] = {}
    for _, row in labels_df.iterrows():
        spec_name = str(row[labels_spec_col]).strip()
        if not spec_name:
            raise ValueError("Empty spec name found in labels.tsv.")
        if spec_name in labels_by_spec:
            raise ValueError(f"Duplicate spec name in labels.tsv: {spec_name}")
        labels_by_spec[spec_name] = {col: str(row[col]) for col in row.index}

    split_rows: list[tuple[str, str]] = []
    for _, row in split_df.iterrows():
        spec_name = str(row[split_spec_col]).strip()
        split_name = str(row[split_name_col]).strip()
        split_rows.append((spec_name, split_name))

    train_specs, query_specs = resolve_split_assignments(split_rows, args.query_splits)
    if not query_specs:
        raise ValueError(
            "No query spectra selected; check labels/split files and --query-splits."
        )

    train_library = _build_library_from_train(
        labels_by_spec=labels_by_spec,
        train_specs=train_specs,
        bits=args.fingerprint_bits,
        radius=args.fingerprint_radius,
        smiles_col=labels_smiles_col,
        inchi_col=labels_inchi_col,
        formula_col=labels_formula_col,
    )
    train_inchi_set = {item.inchi_key_first_block for item in train_library}

    metadata_df = pd.read_csv(metadata_path)
    if "spec_name" not in metadata_df.columns:
        raise ValueError("Metadata CSV must contain a spec_name column.")

    with np.load(fp_path, mmap_mode="r") as loaded:
        if "mist_binary" not in loaded.files:
            raise ValueError("fingerprints.npz must contain mist_binary array.")
        mist_binary = np.asarray(loaded["mist_binary"], dtype=np.float32)

    if mist_binary.ndim != 2:
        raise ValueError("mist_binary must be a 2D array [num_rows, bits].")
    if mist_binary.shape[1] != args.fingerprint_bits:
        raise ValueError(
            "mist_binary width does not match --fingerprint-bits: "
            f"{mist_binary.shape[1]} != {args.fingerprint_bits}"
        )

    metadata_index: dict[str, int] = {}
    for index, row in metadata_df.iterrows():
        spec_name = str(row["spec_name"]).strip()
        if spec_name in metadata_index:
            raise ValueError(f"Duplicate spec name in metadata.csv: {spec_name}")
        metadata_index[spec_name] = index

    query_specs = select_exported_query_specs(
        list(metadata_index),
        set(query_specs),
    )

    detailed_rows: list[dict[str, Any]] = []
    predictions_rows: list[dict[str, Any]] = []
    candidate_score_rows: list[dict[str, Any]] = []
    query_inchi_blocks: list[str] = []

    query_payloads: list[dict[str, Any]] = []
    for query_spec_name in query_specs:
        if query_spec_name not in labels_by_spec:
            raise ValueError(f"Query spec missing from labels.tsv: {query_spec_name}")
        if query_spec_name not in metadata_index:
            raise ValueError(
                f"Query spec missing from metadata.csv / fingerprints.npz: {query_spec_name}"
            )

        label_row = labels_by_spec[query_spec_name]
        query_smiles = label_row.get(labels_smiles_col, "").strip()
        if not query_smiles:
            raise ValueError(f"Missing smiles for query spec: {query_spec_name}")
        query_inchi = str(label_row.get(labels_inchi_col, "")).strip()
        if not query_inchi:
            raise ValueError(f"Missing InChIKey for query spec: {query_spec_name}")
        query_inchi_block = query_inchi.split("-")[0]

        query_formula_raw = label_row.get(labels_formula_col, "").strip()
        query_formula = normalize_formula(query_formula_raw) if query_formula_raw else ""
        query_inchi_blocks.append(query_inchi_block)
        query_payloads.append(
            {
                "spec_name": query_spec_name,
                "smiles": query_smiles,
                "inchi_key_first_block": query_inchi_block,
                "formula": query_formula,
                "query_fp": np.asarray(
                    mist_binary[metadata_index[query_spec_name]], dtype=np.float32
                ),
            }
        )

    validate_query_leakage(query_inchi_blocks, train_inchi_set, args.allow_query_in_train)

    for payload in query_payloads:
        query_spec_name = payload["spec_name"]
        query_smiles = payload["smiles"]
        query_inchi_block = payload["inchi_key_first_block"]
        query_formula = payload["formula"]
        query_fp = payload["query_fp"]

        ranked = rank_train_candidates(
            query_fp=query_fp,
            query_formula=query_formula,
            train_library=train_library,
            top_k=args.top_k,
        )

        query_target_fp = compute_morgan_fingerprint(
            query_smiles,
            args.fingerprint_bits,
            args.fingerprint_radius,
        )
        metrics = evaluate_ranked_predictions(
            target_inchi_key_first_block=query_inchi_block,
            target_fingerprint=query_target_fp,
            ranked=ranked,
            top_k=args.top_k,
        )
        mist_tanimoto = (
            compute_tanimoto_similarity(query_target_fp, query_fp)
            if query_target_fp is not None
            else 0.0
        )

        detailed_rows.append(
            {
                "spec_name": query_spec_name,
                "fingerprint_source": "train_retrieval",
                "target_smiles": query_smiles,
                "target_inchi_key": query_inchi_block,
                "target_formula": query_formula,
                "mist_tanimoto": mist_tanimoto,
                "num_ranked_candidates": len(ranked),
                "exact_match_top1": metrics["exact_match_top1"],
                "exact_match_top10": metrics["exact_match_top10"],
                "tanimoto_top1": metrics["tanimoto_top1"],
                "tanimoto_top10": metrics["tanimoto_top10"],
                "total_formula_matched": metrics["formula_match_count"],
                "total_valid": len(ranked),
                "total_generated": len(train_library),
            }
        )

        row = {"true_smiles": query_smiles, "name": query_spec_name}
        for index in range(args.top_k):
            key = f"pred_smiles_{index + 1}"
            if index < len(ranked):
                row[key] = ranked[index].candidate.smiles
            else:
                row[key] = ""
        predictions_rows.append(row)

        for ranked_entry in ranked:
            candidate_score_rows.append(
                {
                    "query_spec_name": query_spec_name,
                    "query_formula": query_formula,
                    "rank": ranked_entry.rank,
                    "candidate_spec_name": ranked_entry.candidate.spec_name,
                    "candidate_smiles": ranked_entry.candidate.smiles,
                    "candidate_inchi_key_first_block": ranked_entry.candidate.inchi_key_first_block,
                    "candidate_formula": ranked_entry.candidate.formula,
                    "candidate_tanimoto": ranked_entry.tanimoto,
                    "candidate_formula_match": int(ranked_entry.formula_match),
                }
            )

    if not detailed_rows:
        raise ValueError("No detailed rows were produced; check input split configuration.")

    pd.DataFrame(detailed_rows).to_csv(output_path / "detailed_results.csv", index=False)
    pd.DataFrame(candidate_score_rows).to_csv(
        output_path / "candidate_scores.csv",
        index=False,
    )
    pd.DataFrame(predictions_rows).to_csv(output_path / "predictions.csv", index=False)

    exact_top1_scores = [row["exact_match_top1"] for row in detailed_rows]
    exact_top10_scores = [row["exact_match_top10"] for row in detailed_rows]
    tanimoto_top1_scores = [row["tanimoto_top1"] for row in detailed_rows]
    tanimoto_top10_scores = [row["tanimoto_top10"] for row in detailed_rows]
    n_queries = len(detailed_rows)

    aggregate = {
        "schema_version": 1,
        "n_queries": n_queries,
        "top_k": args.top_k,
        "fingerprint_bits": args.fingerprint_bits,
        "fingerprint_radius": args.fingerprint_radius,
        "query_splits": list(args.query_splits),
        "allow_query_in_train": args.allow_query_in_train,
        "train_library_size": len(train_library),
        "exact_match_top1": float(np.mean(exact_top1_scores)),
        "exact_match_top10": float(np.mean(exact_top10_scores)),
        "tanimoto_top1_mean": float(np.mean(tanimoto_top1_scores)),
        "tanimoto_top10_mean": float(np.mean(tanimoto_top10_scores)),
    }
    with (output_path / "aggregate_statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)

    run_manifest = {
        "schema_version": 1,
        "script": os.path.abspath(__file__),
        "labels_tsv": {
            "path": str(labels_path),
            "sha256": _sha256_hex(labels_path),
        },
        "split_tsv": {
            "path": str(split_path),
            "sha256": _sha256_hex(split_path),
        },
        "mist_metadata_csv": {
            "path": str(metadata_path),
            "sha256": _sha256_hex(metadata_path),
        },
        "mist_fingerprints_npz": {
            "path": str(fp_path),
            "sha256": _sha256_hex(fp_path),
        },
        "parameters": vars(args),
        "outputs": {
            "candidate_scores_csv": str(output_path / "candidate_scores.csv"),
            "predictions_csv": str(output_path / "predictions.csv"),
            "detailed_results_csv": str(output_path / "detailed_results.csv"),
            "aggregate_statistics_json": str(output_path / "aggregate_statistics.json"),
        },
    }
    with (output_path / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
