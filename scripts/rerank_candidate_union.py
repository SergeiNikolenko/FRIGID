#!/usr/bin/env python
"""Evaluate target-blind formula and cross-source consensus reranking."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fuse_candidate_sources import (
    RankedCandidate,
    SourceCandidate,
    _dedupe_key,
    _load_metadata,
    _load_source_rows,
    _parse_source_argument,
    _rank_key,
    evaluate_ranked_predictions,
)
from retrieve_train_candidates import (
    _sha256_hex,
    compute_morgan_fingerprint,
    compute_tanimoto_similarity,
)


@dataclass(frozen=True)
class RerankConfig:
    name: str
    formula_bonus: float
    consensus_bonus: float


# This family is deliberately fixed before looking at the locked holdout.
RERANK_CONFIGS = (
    RerankConfig("mist_only", 0.0, 0.0),
    RerankConfig("formula_0p01", 0.01, 0.0),
    RerankConfig("formula_0p025", 0.025, 0.0),
    RerankConfig("formula_0p05", 0.05, 0.0),
    RerankConfig("consensus_0p02", 0.0, 0.02),
    RerankConfig("consensus_0p05", 0.0, 0.05),
    RerankConfig("consensus_0p1", 0.0, 0.1),
    RerankConfig("formula_0p01_consensus_0p02", 0.01, 0.02),
    RerankConfig("formula_0p025_consensus_0p05", 0.025, 0.05),
    RerankConfig("formula_0p05_consensus_0p1", 0.05, 0.1),
)

BASELINE_CONFIG = RERANK_CONFIGS[0]


@dataclass(frozen=True)
class CandidateFeatures:
    representative: SourceCandidate
    candidate_formula: str
    formula_match: bool
    source_support_count: int
    source_support_fraction: float
    cross_source_neighborhood: float
    consensus_signal: float
    tanimoto_to_mist: float


@dataclass(frozen=True)
class RerankedCandidate:
    ranked: RankedCandidate
    candidate_formula: str
    formula_match: bool
    source_support_count: int
    source_support_fraction: float
    cross_source_neighborhood: float
    consensus_signal: float
    rerank_score: float


def compute_candidate_formula(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
    except Exception as exc:
        raise RuntimeError("RDKit is required to compute molecular formulae.") from exc

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Could not parse candidate SMILES: {smiles!r}")
    return str(rdMolDescriptors.CalcMolFormula(molecule))


def load_manifest_formulas(
    manifest_path: Path,
) -> tuple[list[str], dict[str, str]]:
    frame = pd.read_csv(manifest_path, sep="\t").fillna("")
    required = {"spec_name", "formula"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Benchmark manifest is missing columns: {missing}")

    query_order: list[str] = []
    formulas: dict[str, str] = {}
    for index, row in frame.iterrows():
        spec_name = str(row["spec_name"]).strip()
        formula = str(row["formula"]).strip()
        if not spec_name:
            raise ValueError(f"Benchmark manifest has empty spec_name at row {index}.")
        if not formula:
            raise ValueError(
                f"Benchmark manifest has empty formula for {spec_name} at row {index}."
            )
        if spec_name in formulas:
            raise ValueError(f"Duplicate spec_name in benchmark manifest: {spec_name}")
        query_order.append(spec_name)
        formulas[spec_name] = formula
    if not query_order:
        raise ValueError("Benchmark manifest contains no queries.")
    return query_order, formulas


def build_candidate_features(
    query_fp: np.ndarray,
    raw_candidates: list[SourceCandidate],
    query_formula: str,
    source_names: list[str],
) -> list[CandidateFeatures]:
    if len(source_names) != len(set(source_names)):
        raise ValueError("Source names must be unique.")

    expected_sources = set(source_names)
    unexpected_sources = sorted(
        {candidate.source_name for candidate in raw_candidates}.difference(
            expected_sources
        )
    )
    if unexpected_sources:
        raise ValueError(f"Candidates use undeclared sources: {unexpected_sources}")

    by_connectivity: dict[str, list[SourceCandidate]] = defaultdict(list)
    by_source: dict[str, list[SourceCandidate]] = defaultdict(list)
    for candidate in raw_candidates:
        by_connectivity[candidate.inchi_key_first_block].append(candidate)
        by_source[candidate.source_name].append(candidate)

    source_count = len(source_names)
    features: list[CandidateFeatures] = []
    for connectivity_rows in by_connectivity.values():
        representative = min(connectivity_rows, key=_dedupe_key)
        support_sources = {candidate.source_name for candidate in connectivity_rows}

        if source_count <= 1:
            support_fraction = 0.0
            neighborhood = 0.0
        else:
            support_fraction = (len(support_sources) - 1) / (source_count - 1)
            per_source_max = []
            for source_name in source_names:
                similarities = [
                    compute_tanimoto_similarity(
                        representative.fingerprint,
                        other.fingerprint,
                    )
                    for other in by_source.get(source_name, [])
                ]
                per_source_max.append(max(similarities, default=0.0))
            # Remove one guaranteed self-match, leaving only independent-source support.
            neighborhood = (sum(per_source_max) - 1.0) / (source_count - 1)
            neighborhood = min(1.0, max(0.0, neighborhood))

        candidate_formula = compute_candidate_formula(representative.smiles)
        consensus_signal = 0.5 * (support_fraction + neighborhood)
        features.append(
            CandidateFeatures(
                representative=representative,
                candidate_formula=candidate_formula,
                formula_match=candidate_formula == query_formula,
                source_support_count=len(support_sources),
                source_support_fraction=float(support_fraction),
                cross_source_neighborhood=float(neighborhood),
                consensus_signal=float(consensus_signal),
                tanimoto_to_mist=compute_tanimoto_similarity(
                    query_fp,
                    representative.fingerprint,
                ),
            )
        )

    return sorted(features, key=lambda row: _dedupe_key(row.representative))


def rerank_candidate_features(
    features: list[CandidateFeatures],
    config: RerankConfig,
    top_k: int,
) -> list[RerankedCandidate]:
    if top_k <= 0:
        raise ValueError("--top-k must be positive.")

    scored: list[tuple[float, CandidateFeatures]] = []
    for feature in features:
        score = (
            feature.tanimoto_to_mist
            + config.formula_bonus * float(feature.formula_match)
            + config.consensus_bonus * feature.consensus_signal
        )
        scored.append((float(score), feature))
    scored.sort(
        key=lambda item: (
            -item[0],
            *_rank_key(item[1].representative, item[1].tanimoto_to_mist),
        )
    )

    ranked: list[RerankedCandidate] = []
    for rank, (score, feature) in enumerate(scored[:top_k], start=1):
        candidate = feature.representative
        ranked.append(
            RerankedCandidate(
                ranked=RankedCandidate(
                    rank=rank,
                    source_name=candidate.source_name,
                    source_rank=candidate.source_rank,
                    source_candidate_rank=candidate.source_rank,
                    candidate_smiles=candidate.smiles,
                    candidate_spec_name=candidate.candidate_spec_name,
                    candidate_inchi_key_first_block=(candidate.inchi_key_first_block),
                    tanimoto_to_mist=feature.tanimoto_to_mist,
                    fingerprint=candidate.fingerprint,
                ),
                candidate_formula=feature.candidate_formula,
                formula_match=feature.formula_match,
                source_support_count=feature.source_support_count,
                source_support_fraction=feature.source_support_fraction,
                cross_source_neighborhood=feature.cross_source_neighborhood,
                consensus_signal=feature.consensus_signal,
                rerank_score=score,
            )
        )
    return ranked


def build_target_blind_rankings(
    query_order: list[str],
    mist_binary: np.ndarray,
    manifest_formulas: dict[str, str],
    source_rows_by_query: dict[str, list[SourceCandidate]],
    source_names: list[str],
    configs: tuple[RerankConfig, ...],
    top_k: int,
) -> dict[str, dict[str, list[RerankedCandidate]]]:
    """Build rankings without accepting any target-derived input."""
    rankings = {config.name: {} for config in configs}
    for query_index, spec_name in enumerate(query_order):
        features = build_candidate_features(
            query_fp=mist_binary[query_index],
            raw_candidates=source_rows_by_query.get(spec_name, []),
            query_formula=manifest_formulas[spec_name],
            source_names=source_names,
        )
        for config in configs:
            rankings[config.name][spec_name] = rerank_candidate_features(
                features,
                config,
                top_k=top_k,
            )
    return rankings


def evaluate_queries(
    spec_names: list[str],
    rankings_by_spec: dict[str, list[RerankedCandidate]],
    targets_by_spec: dict[str, dict[str, str]],
    top_k: int,
    fingerprint_bits: int,
    fingerprint_radius: int,
) -> dict[str, float]:
    metric_names = (
        "exact_match_top1",
        "exact_match_top10",
        "tanimoto_top1",
        "tanimoto_top10",
    )
    values = {metric_name: [] for metric_name in metric_names}
    for spec_name in spec_names:
        target = targets_by_spec[spec_name]
        target_fingerprint = compute_morgan_fingerprint(
            target["target_smiles"],
            bits=fingerprint_bits,
            radius=fingerprint_radius,
        )
        metrics = evaluate_ranked_predictions(
            target_inchi_key_first_block=target["target_inchi_key"],
            target_fingerprint=target_fingerprint,
            ranked=[entry.ranked for entry in rankings_by_spec[spec_name]],
            top_k=top_k,
        )
        for metric_name in metric_names:
            values[metric_name].append(metrics[metric_name])

    return {
        "n_queries": len(spec_names),
        **{
            metric_name: float(np.mean(metric_values)) if metric_values else 0.0
            for metric_name, metric_values in values.items()
        },
    }


def selection_objective(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (
        metrics["tanimoto_top10"],
        metrics["tanimoto_top1"],
        metrics["exact_match_top10"],
    )


def select_dev_configuration(
    metrics_by_config: dict[str, dict[str, float]],
    configs: tuple[RerankConfig, ...] = RERANK_CONFIGS,
) -> RerankConfig:
    candidates = [config for config in configs if config != BASELINE_CONFIG]
    if not candidates:
        raise ValueError("At least one non-baseline configuration is required.")

    selected = candidates[0]
    for config in candidates[1:]:
        if selection_objective(metrics_by_config[config.name]) > selection_objective(
            metrics_by_config[selected.name]
        ):
            selected = config
    return selected


def _metrics_delta(
    candidate: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float]:
    return {
        metric_name: float(candidate[metric_name] - baseline[metric_name])
        for metric_name in (
            "exact_match_top1",
            "exact_match_top10",
            "tanimoto_top1",
            "tanimoto_top10",
        )
    }


def _ranking_rows(
    query_order: list[str],
    rankings_by_spec: dict[str, list[RerankedCandidate]],
    config: RerankConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec_name in query_order:
        for entry in rankings_by_spec[spec_name]:
            ranked = entry.ranked
            rows.append(
                {
                    "query_spec_name": spec_name,
                    "rank": ranked.rank,
                    "source_name": ranked.source_name,
                    "source_rank": ranked.source_rank,
                    "candidate_smiles": ranked.candidate_smiles,
                    "candidate_spec_name": ranked.candidate_spec_name or "",
                    "candidate_inchi_key_first_block": (
                        ranked.candidate_inchi_key_first_block
                    ),
                    "candidate_formula": entry.candidate_formula,
                    "formula_match": int(entry.formula_match),
                    "source_support_count": entry.source_support_count,
                    "source_support_fraction": entry.source_support_fraction,
                    "cross_source_neighborhood": (entry.cross_source_neighborhood),
                    "consensus_signal": entry.consensus_signal,
                    "tanimoto_to_mist": ranked.tanimoto_to_mist,
                    "rerank_score": entry.rerank_score,
                    "formula_bonus": config.formula_bonus,
                    "consensus_bonus": config.consensus_bonus,
                }
            )
    return rows


def run_reranker_evaluation(
    source_specs: list[tuple[str, Path]],
    mist_metadata_csv: Path,
    mist_fingerprints_npz: Path,
    benchmark_manifest: Path,
    output_dir: Path,
    selection_count: int,
    top_k: int,
    fingerprint_bits: int,
    fingerprint_radius: int,
) -> dict[str, Any]:
    metadata_order, targets_by_spec = _load_metadata(mist_metadata_csv)
    manifest_order, manifest_formulas = load_manifest_formulas(benchmark_manifest)
    if metadata_order != manifest_order:
        raise ValueError(
            "MIST metadata order must exactly match the explicit benchmark manifest."
        )
    if selection_count <= 0 or selection_count >= len(manifest_order):
        raise ValueError(
            "--selection-count must leave non-empty dev and locked holdout splits."
        )

    with np.load(mist_fingerprints_npz, mmap_mode="r") as loaded:
        if "mist_binary" not in loaded.files:
            raise ValueError("fingerprints.npz must contain mist_binary.")
        mist_binary = np.asarray(loaded["mist_binary"], dtype=np.float32)
    expected_shape = (len(manifest_order), fingerprint_bits)
    if mist_binary.shape != expected_shape:
        raise ValueError(
            f"mist_binary shape must be {expected_shape}; got {mist_binary.shape}."
        )

    source_names = [name for name, _ in source_specs]
    if len(source_names) != len(set(source_names)):
        raise ValueError("Duplicate source names are not allowed.")

    source_rows_by_query: dict[str, list[SourceCandidate]] = defaultdict(list)
    source_row_counts: dict[str, int] = {}
    known_queries = set(manifest_order)
    for source_priority, (source_name, source_path) in enumerate(source_specs):
        rows = _load_source_rows(
            source_name=source_name,
            source_path=source_path,
            source_priority=source_priority,
            fingerprint_bits=fingerprint_bits,
            fingerprint_radius=fingerprint_radius,
        )
        unknown_queries = sorted(
            {row.query_spec_name for row in rows}.difference(known_queries)
        )
        if unknown_queries:
            raise ValueError(
                f"{source_name} contains queries absent from manifest: "
                f"{unknown_queries[:3]}"
            )
        source_row_counts[source_name] = len(rows)
        for row in rows:
            source_rows_by_query[row.query_spec_name].append(row)

    rankings = build_target_blind_rankings(
        query_order=manifest_order,
        mist_binary=mist_binary,
        manifest_formulas=manifest_formulas,
        source_rows_by_query=source_rows_by_query,
        source_names=source_names,
        configs=RERANK_CONFIGS,
        top_k=top_k,
    )

    dev_specs = manifest_order[:selection_count]
    holdout_specs = manifest_order[selection_count:]
    dev_metrics: dict[str, dict[str, float]] = {}
    for config in RERANK_CONFIGS:
        dev_metrics[config.name] = evaluate_queries(
            spec_names=dev_specs,
            rankings_by_spec=rankings[config.name],
            targets_by_spec=targets_by_spec,
            top_k=top_k,
            fingerprint_bits=fingerprint_bits,
            fingerprint_radius=fingerprint_radius,
        )

    selected_config = select_dev_configuration(dev_metrics)
    baseline_holdout = evaluate_queries(
        spec_names=holdout_specs,
        rankings_by_spec=rankings[BASELINE_CONFIG.name],
        targets_by_spec=targets_by_spec,
        top_k=top_k,
        fingerprint_bits=fingerprint_bits,
        fingerprint_radius=fingerprint_radius,
    )
    selected_holdout = evaluate_queries(
        spec_names=holdout_specs,
        rankings_by_spec=rankings[selected_config.name],
        targets_by_spec=targets_by_spec,
        top_k=top_k,
        fingerprint_bits=fingerprint_bits,
        fingerprint_radius=fingerprint_radius,
    )
    holdout_positive = selection_objective(selected_holdout) > selection_objective(
        baseline_holdout
    )

    full_baseline = evaluate_queries(
        spec_names=manifest_order,
        rankings_by_spec=rankings[BASELINE_CONFIG.name],
        targets_by_spec=targets_by_spec,
        top_k=top_k,
        fingerprint_bits=fingerprint_bits,
        fingerprint_radius=fingerprint_radius,
    )
    full_selected = evaluate_queries(
        spec_names=manifest_order,
        rankings_by_spec=rankings[selected_config.name],
        targets_by_spec=targets_by_spec,
        top_k=top_k,
        fingerprint_bits=fingerprint_bits,
        fingerprint_radius=fingerprint_radius,
    )

    dev_rows = []
    for config in RERANK_CONFIGS:
        dev_rows.append(
            {
                **asdict(config),
                **dev_metrics[config.name],
                "selected": int(config == selected_config),
            }
        )

    result = {
        "schema_version": 1,
        "selection_protocol": {
            "dev": f"ordered first {selection_count}",
            "locked_holdout": f"ordered last {len(holdout_specs)}",
            "objective": [
                "tanimoto_top10",
                "tanimoto_top1",
                "exact_match_top10",
            ],
            "holdout_configs_evaluated": [
                BASELINE_CONFIG.name,
                selected_config.name,
            ],
        },
        "selected_config": asdict(selected_config),
        "dev_selected_metrics": dev_metrics[selected_config.name],
        "dev_baseline_metrics": dev_metrics[BASELINE_CONFIG.name],
        "locked_holdout": {
            "baseline": baseline_holdout,
            "selected": selected_holdout,
            "delta": _metrics_delta(selected_holdout, baseline_holdout),
            "positive_by_declared_objective": holdout_positive,
        },
        "full_baseline": full_baseline,
        "full_selected": full_selected,
        "decision": "promote_to_diverse_200" if holdout_positive else "reject",
        "leakage_contract": {
            "ranking_inputs": [
                "mist_binary",
                "source_rank_and_order",
                "candidate_smiles_and_morgan_fingerprint",
                "source_identity_and_cross_source_agreement",
                "query_formula_from_explicit_benchmark_manifest",
            ],
            "candidate_formula_source": "RDKit CalcMolFormula(candidate_smiles)",
            "target_fields_used_only_after_rankings_are_built": [
                "target_smiles",
                "target_inchi_key",
                "target_fingerprint",
            ],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dev_rows).to_csv(output_dir / "dev_selection.csv", index=False)
    pd.DataFrame(
        _ranking_rows(
            manifest_order,
            rankings[selected_config.name],
            selected_config,
        )
    ).to_csv(output_dir / "selected_prediction_scores.csv", index=False)
    pd.DataFrame(
        _ranking_rows(
            manifest_order,
            rankings[BASELINE_CONFIG.name],
            BASELINE_CONFIG,
        )
    ).to_csv(output_dir / "mist_only_prediction_scores.csv", index=False)
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    source_payload = [
        {
            "name": source_name,
            "path": str(source_path),
            "sha256": _sha256_hex(source_path),
            "row_count": source_row_counts[source_name],
        }
        for source_name, source_path in source_specs
    ]
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "script": str(Path(__file__).resolve()),
                "sources": source_payload,
                "mist_metadata_csv": {
                    "path": str(mist_metadata_csv),
                    "sha256": _sha256_hex(mist_metadata_csv),
                },
                "mist_fingerprints_npz": {
                    "path": str(mist_fingerprints_npz),
                    "sha256": _sha256_hex(mist_fingerprints_npz),
                },
                "benchmark_manifest": {
                    "path": str(benchmark_manifest),
                    "sha256": _sha256_hex(benchmark_manifest),
                },
                "parameters": {
                    "selection_count": selection_count,
                    "top_k": top_k,
                    "fingerprint_bits": fingerprint_bits,
                    "fingerprint_radius": fingerprint_radius,
                    "predeclared_configs": [
                        asdict(config) for config in RERANK_CONFIGS
                    ],
                },
                "outputs": {
                    "evaluation_json": str(output_dir / "evaluation.json"),
                    "dev_selection_csv": str(output_dir / "dev_selection.csv"),
                    "selected_prediction_scores_csv": str(
                        output_dir / "selected_prediction_scores.csv"
                    ),
                    "mist_only_prediction_scores_csv": str(
                        output_dir / "mist_only_prediction_scores.csv"
                    ),
                },
            },
            handle,
            indent=2,
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a target-blind formula/consensus reranker on an ordered dev "
            "split and evaluate it once on the locked holdout."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        type=_parse_source_argument,
        help="Repeated candidate source in NAME=PATH form.",
    )
    parser.add_argument("--mist-metadata-csv", required=True)
    parser.add_argument("--mist-fingerprints-npz", required=True)
    parser.add_argument(
        "--benchmark-manifest",
        required=True,
        help="Explicit ordered TSV containing spec_name and query formula.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-count", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--fingerprint-bits", type=int, default=4096)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_reranker_evaluation(
        source_specs=args.source,
        mist_metadata_csv=Path(args.mist_metadata_csv).expanduser().resolve(),
        mist_fingerprints_npz=Path(args.mist_fingerprints_npz).expanduser().resolve(),
        benchmark_manifest=Path(args.benchmark_manifest).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        selection_count=args.selection_count,
        top_k=args.top_k,
        fingerprint_bits=args.fingerprint_bits,
        fingerprint_radius=args.fingerprint_radius,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
