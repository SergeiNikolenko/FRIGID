#!/usr/bin/env python
"""Calibrate target-blind conformal candidate sets for a frozen RankLoop ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QueryCandidates:
    spec_name: str
    target_id: str
    candidate_ids: tuple[str, ...]
    candidate_smiles: tuple[str, ...]
    scores: np.ndarray

    @property
    def target_index(self) -> int | None:
        try:
            return self.candidate_ids.index(self.target_id)
        except ValueError:
            return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalize_inchi_key(value: Any) -> str:
    return str(value).strip().split("-", maxsplit=1)[0]


def load_manifest(path: Path) -> pd.DataFrame:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    frame = pd.read_csv(path, sep=delimiter, dtype=str).fillna("")
    required = {"spec_name", "inchikey_first_block"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Manifest {path} is missing columns: {missing}")
    if frame.empty or frame["spec_name"].duplicated().any():
        raise ValueError(f"Manifest spec_name values must be non-empty and unique: {path}")
    if (frame["spec_name"].str.strip() == "").any():
        raise ValueError(f"Manifest contains an empty spec_name: {path}")
    return frame


def load_queries(
    candidates_path: Path,
    targets_path: Path,
    manifest_path: Path,
    score_column: str,
) -> tuple[list[QueryCandidates], pd.DataFrame]:
    candidates = pd.read_csv(candidates_path).fillna("")
    targets = pd.read_csv(targets_path).fillna("")
    manifest = load_manifest(manifest_path)

    candidate_columns = {
        "query_spec_name",
        "rank",
        "candidate_smiles",
        "candidate_inchi_key_first_block",
        score_column,
    }
    missing_candidates = sorted(candidate_columns.difference(candidates.columns))
    if missing_candidates:
        raise ValueError(
            f"Candidate table {candidates_path} is missing columns: {missing_candidates}"
        )
    target_columns = {"spec_name", "target_inchi_key"}
    missing_targets = sorted(target_columns.difference(targets.columns))
    if missing_targets:
        raise ValueError(f"Target table {targets_path} is missing columns: {missing_targets}")

    targets = targets.drop_duplicates("spec_name", keep=False).copy()
    target_map = {
        str(row.spec_name): normalize_inchi_key(row.target_inchi_key)
        for row in targets.itertuples(index=False)
    }
    manifest_order = manifest["spec_name"].astype(str).tolist()
    manifest_targets = {
        str(row.spec_name): normalize_inchi_key(row.inchikey_first_block)
        for row in manifest.itertuples(index=False)
    }
    if set(target_map) != set(manifest_order):
        missing = sorted(set(manifest_order).difference(target_map))
        extra = sorted(set(target_map).difference(manifest_order))
        raise ValueError(f"Target/manifest query mismatch: missing={missing[:5]}, extra={extra[:5]}")
    mismatched_targets = [
        name for name in manifest_order if target_map[name] != manifest_targets[name]
    ]
    if mismatched_targets:
        raise ValueError(f"Target identity disagrees with manifest: {mismatched_targets[:5]}")

    candidates["query_spec_name"] = candidates["query_spec_name"].astype(str)
    candidate_queries = set(candidates["query_spec_name"])
    if candidate_queries != set(manifest_order):
        missing = sorted(set(manifest_order).difference(candidate_queries))
        extra = sorted(candidate_queries.difference(manifest_order))
        raise ValueError(
            f"Candidate/manifest query mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )

    grouped = {name: group for name, group in candidates.groupby("query_spec_name")}
    queries: list[QueryCandidates] = []
    for name in manifest_order:
        group = grouped[name].copy()
        group["rank"] = pd.to_numeric(group["rank"], errors="raise").astype(int)
        group[score_column] = pd.to_numeric(group[score_column], errors="raise")
        group.sort_values(
            ["rank", "candidate_inchi_key_first_block"], kind="stable", inplace=True
        )
        expected_ranks = list(range(1, len(group) + 1))
        if group["rank"].tolist() != expected_ranks:
            raise ValueError(f"Frozen ranks are not contiguous for {name}")
        candidate_ids = tuple(
            normalize_inchi_key(value)
            for value in group["candidate_inchi_key_first_block"]
        )
        if not all(candidate_ids) or len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"Candidate identities must be non-empty and unique for {name}")
        scores = group[score_column].to_numpy(dtype=float)
        if not np.isfinite(scores).all():
            raise ValueError(f"Candidate scores must be finite for {name}")
        if np.any(np.diff(scores) > 1e-12):
            raise ValueError(f"Frozen rank and score order disagree for {name}")
        queries.append(
            QueryCandidates(
                spec_name=name,
                target_id=target_map[name],
                candidate_ids=candidate_ids,
                candidate_smiles=tuple(group["candidate_smiles"].astype(str)),
                scores=scores,
            )
        )
    return queries, manifest


def deterministic_split(
    names: list[str], fit_fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("temperature_fit_fraction must be between zero and one")
    ordered = sorted(
        names,
        key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest(),
    )
    fit_size = min(len(ordered) - 1, max(1, round(len(ordered) * fit_fraction)))
    return set(ordered[:fit_size]), set(ordered[fit_size:])


def softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    logits = scores / temperature
    logits = logits - float(np.max(logits))
    values = np.exp(logits)
    return values / float(values.sum())


def fit_temperature(
    queries: list[QueryCandidates], temperatures: list[float], minimum_targets: int
) -> dict:
    present = [query for query in queries if query.target_index is not None]
    if len(present) < minimum_targets:
        raise ValueError(
            f"Temperature fit has {len(present)} target-present queries; "
            f"requires at least {minimum_targets}"
        )
    rows = []
    for temperature in temperatures:
        losses = []
        for query in present:
            probabilities = softmax(query.scores, temperature)
            losses.append(-math.log(max(float(probabilities[query.target_index]), 1e-300)))
        rows.append({"temperature": float(temperature), "mean_nll": float(np.mean(losses))})
    selected = min(rows, key=lambda row: (row["mean_nll"], row["temperature"]))
    return {
        "selected_temperature": selected["temperature"],
        "target_present_count": len(present),
        "grid": rows,
    }


def nonconformity_scores(
    query: QueryCandidates,
    temperature: float,
    raps_lambda: float = 0.0,
    raps_k_reg: int = 0,
) -> np.ndarray:
    probabilities = softmax(query.scores, temperature)
    ranks = np.arange(1, len(probabilities) + 1)
    penalty = raps_lambda * np.maximum(ranks - raps_k_reg, 0)
    return np.cumsum(probabilities) + penalty


def finite_sample_quantile(scores: list[float], alpha: float) -> dict:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if not scores:
        raise ValueError("At least one conformity score is required")
    ordered = np.sort(np.asarray(scores, dtype=float))
    rank = min(len(ordered), math.ceil((len(ordered) + 1) * (1.0 - alpha)))
    return {
        "value": float(ordered[rank - 1]),
        "sample_count": len(ordered),
        "finite_sample_rank": rank,
        "nominal_coverage": 1.0 - alpha,
    }


def query_features(query: QueryCandidates, temperature: float) -> dict:
    probabilities = softmax(query.scores, temperature)
    margin = float(query.scores[0] - query.scores[1]) if len(query.scores) > 1 else 1.0
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300))))
    normalized_entropy = entropy / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
    return {
        "top_probability": float(probabilities[0]),
        "score_margin": margin,
        "probability_entropy_normalized": normalized_entropy,
        "candidate_count": len(query.candidate_ids),
        "top1_correct": query.target_index == 0,
        "target_present": query.target_index is not None,
    }


def confidence_boundary(
    queries: list[QueryCandidates], temperature: float, quantile: float
) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError("confidence_group_quantile must be between zero and one")
    margins = [query_features(query, temperature)["score_margin"] for query in queries]
    return float(np.quantile(margins, quantile))


def confidence_group(query: QueryCandidates, boundary: float, temperature: float) -> str:
    margin = query_features(query, temperature)["score_margin"]
    return "high_margin" if margin >= boundary else "low_margin"


def calibrate_quantiles(
    queries: list[QueryCandidates],
    temperature: float,
    alpha: float,
    boundary: float,
    minimum_targets: int,
    minimum_group_targets: int,
    raps_lambda: float,
    raps_k_reg: int,
) -> dict:
    present = [query for query in queries if query.target_index is not None]
    if len(present) < minimum_targets:
        raise ValueError(
            f"Conformal calibration has {len(present)} target-present queries; "
            f"requires at least {minimum_targets}"
        )
    methods = {
        "aps": {"raps_lambda": 0.0, "raps_k_reg": 0},
        "raps": {"raps_lambda": raps_lambda, "raps_k_reg": raps_k_reg},
    }
    output = {}
    for method, settings in methods.items():
        by_group: dict[str, list[float]] = {"low_margin": [], "high_margin": []}
        all_scores = []
        for query in present:
            scores = nonconformity_scores(query, temperature, **settings)
            target_score = float(scores[query.target_index])
            all_scores.append(target_score)
            by_group[confidence_group(query, boundary, temperature)].append(target_score)
        global_quantile = finite_sample_quantile(all_scores, alpha)
        group_quantiles = {}
        for group, values in by_group.items():
            if len(values) >= minimum_group_targets:
                group_quantiles[group] = {
                    **finite_sample_quantile(values, alpha),
                    "fallback_to_global": False,
                }
            else:
                group_quantiles[group] = {
                    **global_quantile,
                    "group_sample_count": len(values),
                    "fallback_to_global": True,
                }
        output[method] = {
            "settings": settings,
            "global": global_quantile,
            "confidence_groups": group_quantiles,
        }
    return output


def prediction_set_size(scores: np.ndarray, threshold: float) -> int:
    crossing = np.flatnonzero(scores >= threshold)
    return int(crossing[0] + 1) if len(crossing) else len(scores)


def calibrate_abstention(
    queries: list[QueryCandidates],
    temperature: float,
    maximum_risk: float,
    minimum_queries: int,
) -> dict:
    records = [query_features(query, temperature) for query in queries]
    thresholds = sorted({float(row["top_probability"]) for row in records})
    curve = []
    for threshold in thresholds:
        selected = [row for row in records if row["top_probability"] >= threshold]
        if not selected:
            continue
        risk = 1.0 - float(np.mean([row["top1_correct"] for row in selected]))
        curve.append(
            {
                "threshold": threshold,
                "accepted_count": len(selected),
                "coverage": len(selected) / len(records),
                "empirical_risk": risk,
            }
        )
    eligible = [
        row
        for row in curve
        if row["accepted_count"] >= minimum_queries
        and row["empirical_risk"] <= maximum_risk
    ]
    if eligible:
        selected = max(eligible, key=lambda row: (row["coverage"], row["threshold"]))
        threshold = selected["threshold"]
    else:
        selected = None
        threshold = None
    return {
        "maximum_calibration_risk": maximum_risk,
        "minimum_queries": minimum_queries,
        "selected": selected,
        "threshold": threshold,
        "curve": curve,
        "guarantee": "empirical development risk only; not a conformal risk guarantee",
    }


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> list[float | None]:
    if total == 0:
        return [None, None]
    rate = successes / total
    denominator = 1.0 + z**2 / total
    center = (rate + z**2 / (2.0 * total)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z**2 / (4.0 * total**2)) / denominator
    return [float(max(0.0, center - radius)), float(min(1.0, center + radius))]


def summarize_policy(frame: pd.DataFrame, nominal_coverage: float) -> dict:
    present = frame[frame["target_present"]]
    conditional_successes = int(present["covered"].sum())
    conditional_total = len(present)
    conditional_coverage = (
        conditional_successes / conditional_total if conditional_total else None
    )
    return {
        "queries": len(frame),
        "candidate_recall": float(frame["target_present"].mean()),
        "unconditional_exact_coverage": float(frame["covered"].mean()),
        "conditional_exact_coverage": conditional_coverage,
        "conditional_exact_coverage_wilson95": wilson_interval(
            conditional_successes, conditional_total
        ),
        "conditional_coverage_gap": conditional_coverage - nominal_coverage
        if conditional_coverage is not None
        else None,
        "target_absent_queries": int((~frame["target_present"]).sum()),
        "mean_set_size": float(frame["set_size"].mean()),
        "median_set_size": float(frame["set_size"].median()),
        "p90_set_size": float(frame["set_size"].quantile(0.9)),
        "full_set_fraction": float(
            (frame["set_size"] == frame["candidate_count"]).mean()
        ),
        "nominal_conditional_coverage": nominal_coverage,
    }


def evaluate(
    queries: list[QueryCandidates],
    temperature: float,
    boundary: float,
    quantiles: dict,
    abstention: dict,
    calibration_target_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    rows = []
    policies = []
    for method in ("aps", "raps"):
        for calibration in ("global", "confidence"):
            policies.append((method, calibration, f"{method}_{calibration}"))
    for query in queries:
        features = query_features(query, temperature)
        group = confidence_group(query, boundary, temperature)
        for method, calibration, policy in policies:
            method_data = quantiles[method]
            threshold_data = (
                method_data["global"]
                if calibration == "global"
                else method_data["confidence_groups"][group]
            )
            scores = nonconformity_scores(
                query, temperature, **method_data["settings"]
            )
            set_size = prediction_set_size(scores, threshold_data["value"])
            target_index = query.target_index
            covered = target_index is not None and target_index < set_size
            rows.append(
                {
                    "spec_name": query.spec_name,
                    "policy": policy,
                    "method": method,
                    "calibration": calibration,
                    "confidence_group": group,
                    "quantile": threshold_data["value"],
                    "group_fallback_to_global": bool(
                        threshold_data.get("fallback_to_global", False)
                    ),
                    "set_size": set_size,
                    "candidate_count": len(query.candidate_ids),
                    "candidate_ids_json": json.dumps(query.candidate_ids[:set_size]),
                    "candidate_smiles_json": json.dumps(
                        query.candidate_smiles[:set_size]
                    ),
                    "target_present": features["target_present"],
                    "covered": covered,
                    "top1_correct": features["top1_correct"],
                    "top_probability": features["top_probability"],
                    "score_margin": features["score_margin"],
                    "probability_entropy_normalized": features[
                        "probability_entropy_normalized"
                    ],
                    "accepted": abstention["threshold"] is not None
                    and features["top_probability"] >= abstention["threshold"],
                    "target_seen_in_calibration": query.target_id
                    in calibration_target_ids,
                }
            )
    metrics = pd.DataFrame(rows)
    set_columns = [
        "spec_name",
        "policy",
        "confidence_group",
        "quantile",
        "set_size",
        "candidate_count",
        "candidate_ids_json",
        "candidate_smiles_json",
        "top_probability",
        "score_margin",
        "probability_entropy_normalized",
        "accepted",
    ]
    candidate_sets = metrics[set_columns].copy()
    summaries = {
        policy: summarize_policy(
            metrics[metrics["policy"] == policy],
            quantiles[method]["global"]["nominal_coverage"],
        )
        for method, _calibration, policy in policies
    }
    strata = []
    for policy, policy_frame in metrics.groupby("policy"):
        for column in ("confidence_group", "target_seen_in_calibration"):
            for value, group_frame in policy_frame.groupby(column):
                summary = summarize_policy(
                    group_frame,
                    quantiles[policy.split("_", maxsplit=1)[0]]["global"][
                        "nominal_coverage"
                    ],
                )
                strata.append(
                    {
                        "policy": policy,
                        "stratum": column,
                        "value": str(value),
                        **summary,
                    }
                )
    abstention_selected = metrics.drop_duplicates("spec_name")
    accepted = abstention_selected[abstention_selected["accepted"]]
    abstention_summary = {
        "threshold": abstention["threshold"],
        "accepted_count": len(accepted),
        "coverage": len(accepted) / len(abstention_selected),
        "empirical_exact_risk": 1.0 - float(accepted["top1_correct"].mean())
        if len(accepted)
        else None,
        "top1_exact_accuracy_without_abstention": float(
            abstention_selected["top1_correct"].mean()
        ),
    }
    return candidate_sets, metrics, {"policies": summaries, "abstention": abstention_summary}, pd.DataFrame(strata)


def distribution_audit(
    calibration: list[QueryCandidates], evaluation: list[QueryCandidates], temperature: float
) -> dict:
    calibration_names = {query.spec_name for query in calibration}
    evaluation_names = {query.spec_name for query in evaluation}
    overlap = sorted(calibration_names.intersection(evaluation_names))
    if overlap:
        raise ValueError(f"Calibration/evaluation query overlap: {overlap[:5]}")
    calibration_targets = {query.target_id for query in calibration}
    evaluation_targets = {query.target_id for query in evaluation}

    def feature_summary(queries: list[QueryCandidates]) -> dict:
        rows = [query_features(query, temperature) for query in queries]
        return {
            key: float(np.mean([row[key] for row in rows]))
            for key in (
                "top_probability",
                "score_margin",
                "probability_entropy_normalized",
                "candidate_count",
                "target_present",
            )
        }

    molecule_overlap = calibration_targets.intersection(evaluation_targets)
    return {
        "query_overlap_count": 0,
        "molecule_overlap_count": len(molecule_overlap),
        "evaluation_molecule_overlap_fraction": len(molecule_overlap)
        / len(evaluation_targets),
        "calibration": feature_summary(calibration),
        "evaluation": feature_summary(evaluation),
        "warnings": [
            "Nominal conformal coverage is conditional on the exact target being present in the frozen candidate pool.",
            "Molecule overlap does not invalidate row-level evaluation, but only a molecule-disjoint panel tests structural shift."
            if molecule_overlap
            else "Evaluation molecules are disjoint from calibration molecules; exchangeability may be weakened by structural shift.",
        ],
    }


def candidate_identity_sha256(queries: list[QueryCandidates]) -> str:
    digest = hashlib.sha256()
    for query in queries:
        for rank, (candidate_id, score) in enumerate(
            zip(query.candidate_ids, query.scores), start=1
        ):
            digest.update(
                f"{query.spec_name}\t{rank}\t{candidate_id}\t{score:.17g}\n".encode()
            )
    return digest.hexdigest()


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def run(args: argparse.Namespace) -> dict:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    score_column = str(config["score_column"])
    calibration, calibration_manifest = load_queries(
        args.calibration_candidates,
        args.calibration_targets,
        args.calibration_manifest,
        score_column,
    )
    evaluation, evaluation_manifest = load_queries(
        args.evaluation_candidates,
        args.evaluation_targets,
        args.evaluation_manifest,
        score_column,
    )
    fit_names, conformal_names = deterministic_split(
        [query.spec_name for query in calibration],
        float(config["temperature_fit_fraction"]),
        int(config["split_seed"]),
    )
    fit_queries = [query for query in calibration if query.spec_name in fit_names]
    conformal_queries = [
        query for query in calibration if query.spec_name in conformal_names
    ]
    temperature_fit = fit_temperature(
        fit_queries,
        [float(value) for value in config["temperature_grid"]],
        int(config["minimum_temperature_targets"]),
    )
    temperature = float(temperature_fit["selected_temperature"])
    boundary = confidence_boundary(
        conformal_queries, temperature, float(config["confidence_group_quantile"])
    )
    quantiles = calibrate_quantiles(
        conformal_queries,
        temperature,
        float(config["alpha"]),
        boundary,
        int(config["minimum_conformal_targets"]),
        int(config["minimum_group_targets"]),
        float(config["raps_lambda"]),
        int(config["raps_k_reg"]),
    )
    abstention = calibrate_abstention(
        conformal_queries,
        temperature,
        float(config["maximum_calibration_risk"]),
        int(config["minimum_abstention_queries"]),
    )
    audit = distribution_audit(calibration, evaluation, temperature)
    candidate_sets, query_metrics, summary, strata = evaluate(
        evaluation,
        temperature,
        boundary,
        quantiles,
        abstention,
        {query.target_id for query in calibration},
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    candidate_sets.to_csv(args.output_dir / "candidate_sets.csv", index=False)
    query_metrics.to_csv(args.output_dir / "query_metrics.csv", index=False)
    strata.to_csv(args.output_dir / "stratified_metrics.csv", index=False)
    calibration_payload = {
        "coverage_scope": config["coverage_scope"],
        "temperature_fit": temperature_fit,
        "temperature_fit_queries": sorted(fit_names),
        "conformal_queries": sorted(conformal_names),
        "confidence_margin_boundary": boundary,
        "quantiles": quantiles,
        "abstention": abstention,
    }
    write_json(args.output_dir / "calibration.json", calibration_payload)
    summary["distribution_audit"] = audit
    summary["ranking_unchanged"] = True
    summary["candidate_recall_is_unconditional_coverage_upper_bound"] = True
    write_json(args.output_dir / "summary.json", summary)

    inputs = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in {
            "config": args.config,
            "calibration_candidates": args.calibration_candidates,
            "calibration_targets": args.calibration_targets,
            "calibration_manifest": args.calibration_manifest,
            "evaluation_candidates": args.evaluation_candidates,
            "evaluation_targets": args.evaluation_targets,
            "evaluation_manifest": args.evaluation_manifest,
        }.items()
    }
    manifest = {
        "schema_version": 1,
        "purpose": "rankloop_conformal_candidate_sets",
        "status": "completed",
        "code_commit": git_commit(),
        "inputs": inputs,
        "candidate_identity": {
            "calibration_sha256": candidate_identity_sha256(calibration),
            "evaluation_sha256": candidate_identity_sha256(evaluation),
        },
        "query_counts": {
            "calibration": len(calibration),
            "temperature_fit": len(fit_queries),
            "conformal": len(conformal_queries),
            "evaluation": len(evaluation),
        },
        "target_fields_used": {
            "temperature_fit": ["target_inchi_key"],
            "conformal_calibration": ["target_inchi_key"],
            "inference": [],
            "metrics_only": ["target_inchi_key", "inchikey_first_block"],
        },
        "underlying_ranking_changed": False,
        "manifest_order": {
            "calibration": calibration_manifest["spec_name"].tolist(),
            "evaluation": evaluation_manifest["spec_name"].tolist(),
        },
        "outputs": {},
    }
    for path in sorted(args.output_dir.iterdir()):
        if path.name == "run_manifest.json":
            continue
        manifest["outputs"][path.name] = {
            "sha256": sha256_file(path),
            "path": str(path.resolve()),
        }
    write_json(args.output_dir / "run_manifest.json", manifest)
    return {"calibration": calibration_payload, "summary": summary, "manifest": manifest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--calibration-candidates", required=True, type=Path)
    parser.add_argument("--calibration-targets", required=True, type=Path)
    parser.add_argument("--calibration-manifest", required=True, type=Path)
    parser.add_argument("--evaluation-candidates", required=True, type=Path)
    parser.add_argument("--evaluation-targets", required=True, type=Path)
    parser.add_argument("--evaluation-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
