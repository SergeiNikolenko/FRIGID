#!/usr/bin/env python
"""Run target-blind constrained STONED and two-switch candidate expansion."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from rdkit.Chem import rdFingerprintGenerator
import selfies as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from frigid.stoned_expansion import (  # noqa: E402
    canonicalize_smiles,
    generate_stoned_candidates,
    stable_seed,
)
from frigid.two_switch import generate_two_switch_neighbors  # noqa: E402


SCHEMA_VERSION = 1
TARGET_COLUMNS = {
    "target_smiles",
    "target_inchi_key",
    "target_inchi_key_connectivity",
    "target_fingerprint",
}
VARIANTS = {
    "A_union": (),
    "B_union_two_switch": ("two_switch_B",),
    "C_union_stoned": ("stoned_C",),
    "D_union_two_switch_stoned": ("two_switch_D", "stoned_D"),
}
FP_BITS = 4096
FP_RADIUS = 2
FP_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=FP_RADIUS, fpSize=FP_BITS
)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def assert_tracked_worktree_clean(repo: Path) -> None:
    for args in (("diff", "--quiet"), ("diff", "--cached", "--quiet")):
        result = subprocess.run(["git", "-C", str(repo), *args], check=False)
        if result.returncode != 0:
            raise RuntimeError("Tracked FRIGID files are dirty; commit before scoring")


def load_fixed_manifest(path: Path, expected_sha256: str | None) -> tuple[list[str], str]:
    actual = sha256_file(path)
    if expected_sha256 and actual != expected_sha256:
        raise ValueError(
            f"Fixed manifest SHA-256 mismatch: {actual} != {expected_sha256}"
        )
    separator = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    frame = pd.read_csv(path, sep=separator, usecols=["spec_name"], dtype=str)
    names = frame["spec_name"].str.strip().tolist()
    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("Fixed manifest spec_name values must be unique and non-empty")
    return names, actual


def load_queries(path: Path, spec_names: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    required = {"spec_name", "formula"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Queries file is missing columns: {sorted(missing)}")
    leaked = TARGET_COLUMNS.intersection(frame.columns)
    if leaked:
        raise ValueError(f"Target fields are forbidden in queries: {sorted(leaked)}")
    frame = frame.set_index("spec_name", verify_integrity=True).loc[spec_names].reset_index()
    if frame["spec_name"].tolist() != spec_names:
        raise ValueError("Queries order does not match fixed manifest")
    return frame


def _resolve_candidate_columns(frame: pd.DataFrame) -> tuple[str, str, str]:
    query = "query_spec_name" if "query_spec_name" in frame else "spec_name"
    smiles = "candidate_smiles" if "candidate_smiles" in frame else "smiles"
    if query not in frame or smiles not in frame or "rank" not in frame:
        raise ValueError(
            "Candidate CSV requires query_spec_name/spec_name, rank, and candidate_smiles/smiles"
        )
    return query, "rank", smiles


def load_candidates(path: Path, source_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path).fillna("")
    query_col, rank_col, smiles_col = _resolve_candidate_columns(frame)
    rows = pd.DataFrame(
        {
            "query_spec_name": frame[query_col].astype(str),
            "rank": pd.to_numeric(frame[rank_col], errors="raise").astype(int),
            "candidate_smiles": frame[smiles_col].astype(str),
            "source_name": source_name,
        }
    )
    return rows.sort_values(["query_spec_name", "rank"], kind="stable")


def _candidate_record(smiles: str) -> dict[str, Any] | None:
    prepared = canonicalize_smiles(smiles)
    if prepared is None:
        return None
    mol, canonical, key, formula = prepared
    fingerprint = FP_GENERATOR.GetFingerprintAsNumPy(mol).astype(np.float32)
    return {
        "candidate_smiles": canonical,
        "candidate_inchi_key_connectivity": key,
        "candidate_formula": formula,
        "fingerprint": fingerprint,
    }


def build_baseline_pool(
    union: pd.DataFrame, spec_names: list[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    pools: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in spec_names}
    for row in union.to_dict(orient="records"):
        spec_name = str(row["query_spec_name"])
        if spec_name not in pools:
            continue
        record = _candidate_record(str(row["candidate_smiles"]))
        if record is None:
            continue
        key = record["candidate_inchi_key_connectivity"]
        existing = pools[spec_name].get(key)
        if existing is None or int(row["rank"]) < existing["source_rank"]:
            pools[spec_name][key] = {
                **record,
                "generator": "union",
                "source_name": "union",
                "source_rank": int(row["rank"]),
            }
    return pools


def select_seeds(
    spec_names: list[str],
    query_formulas: dict[str, str],
    source_frames: list[tuple[str, pd.DataFrame]],
    *,
    seeds_per_source: int,
    max_seeds: int,
) -> tuple[pd.DataFrame, dict[str, set[str]], Counter[str]]:
    rows: list[dict[str, Any]] = []
    original_keys: dict[str, set[str]] = {name: set() for name in spec_names}
    rejected: Counter[str] = Counter()
    by_query_source: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for source_name, frame in source_frames:
        for spec_name in spec_names:
            candidates: list[dict[str, Any]] = []
            query_rows = frame[frame["query_spec_name"] == spec_name]
            for row in query_rows.to_dict(orient="records"):
                record = _candidate_record(str(row["candidate_smiles"]))
                if record is None:
                    rejected["invalid_seed"] += 1
                    continue
                original_keys[spec_name].add(record["candidate_inchi_key_connectivity"])
                if record["candidate_formula"] != query_formulas[spec_name]:
                    rejected["formula_mismatch_seed"] += 1
                    continue
                candidates.append(
                    {
                        **record,
                        "source_name": source_name,
                        "source_rank": int(row["rank"]),
                    }
                )
            by_query_source[(spec_name, source_name)] = candidates

    for spec_name in spec_names:
        selected_keys: set[str] = set()
        selected: list[dict[str, Any]] = []
        positions = {source_name: 0 for source_name, _ in source_frames}
        selected_per_source = Counter()
        while len(selected) < max_seeds:
            made_progress = False
            for source_name, _ in source_frames:
                candidates = by_query_source[(spec_name, source_name)]
                if selected_per_source[source_name] >= seeds_per_source:
                    continue
                while positions[source_name] < len(candidates):
                    row = candidates[positions[source_name]]
                    positions[source_name] += 1
                    key = row["candidate_inchi_key_connectivity"]
                    if key in selected_keys:
                        rejected["duplicate_seed"] += 1
                        continue
                    selected_keys.add(key)
                    selected.append(row)
                    selected_per_source[source_name] += 1
                    made_progress = True
                    break
                if len(selected) >= max_seeds:
                    break
            if not made_progress:
                break
        if not selected:
            raise ValueError(f"No formula-valid seeds for {spec_name}")
        for index, row in enumerate(selected, start=1):
            rows.append(
                {
                    "query_spec_name": spec_name,
                    "seed_index": index,
                    "seed_smiles": row["candidate_smiles"],
                    "seed_inchi_key_connectivity": row[
                        "candidate_inchi_key_connectivity"
                    ],
                    "seed_source": row["source_name"],
                    "seed_source_rank": row["source_rank"],
                    "query_formula": query_formulas[spec_name],
                }
            )
    return pd.DataFrame(rows), original_keys, rejected


def generate_expansions(
    seeds: pd.DataFrame,
    original_keys: dict[str, set[str]],
    *,
    global_seed: int,
    raw_proposals_per_seed: int,
    accepted_per_seed: int,
    mutation_depths: tuple[int, ...],
    operations: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    if raw_proposals_per_seed % 2 or accepted_per_seed % 2:
        raise ValueError("Combined D budget requires even proposal and accepted budgets")
    raw_rows: list[dict[str, Any]] = []
    accepted_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    rejection_totals: Counter[str] = Counter()
    generator_stats: dict[str, Counter[str]] = defaultdict(Counter)

    generation_plan = (
        ("stoned_C", "stoned", raw_proposals_per_seed, accepted_per_seed),
        ("two_switch_B", "two_switch", raw_proposals_per_seed, accepted_per_seed),
        ("stoned_D", "stoned", raw_proposals_per_seed // 2, accepted_per_seed // 2),
        (
            "two_switch_D",
            "two_switch",
            raw_proposals_per_seed // 2,
            accepted_per_seed // 2,
        ),
    )

    for seed_row in seeds.to_dict(orient="records"):
        spec_name = seed_row["query_spec_name"]
        for role, generator, proposal_budget, accepted_budget in generation_plan:
            per_seed = stable_seed(
                global_seed,
                spec_name,
                seed_row["seed_inchi_key_connectivity"],
                generator,
            )
            started = time.perf_counter()
            if generator == "stoned":
                candidates, proposals, stats = generate_stoned_candidates(
                    seed_row["seed_smiles"],
                    query_formula=seed_row["query_formula"],
                    seed=per_seed,
                    raw_proposals=proposal_budget,
                    mutation_depths=mutation_depths,
                    operations=operations,
                    max_accepted=accepted_budget,
                    exclude_connectivity_keys=original_keys[spec_name],
                )
                for proposal in proposals:
                    raw_rows.append(
                        {
                            "query_spec_name": spec_name,
                            "comparison_role": role,
                            "seed_smiles": seed_row["seed_smiles"],
                            "seed_inchi_key_connectivity": seed_row[
                                "seed_inchi_key_connectivity"
                            ],
                            "seed_source": seed_row["seed_source"],
                            "rng_seed": per_seed,
                            **proposal.as_dict(),
                        }
                    )
                for candidate in candidates:
                    accepted_rows.append(
                        {
                            "query_spec_name": spec_name,
                            "generator": role,
                            "seed_smiles": seed_row["seed_smiles"],
                            "seed_inchi_key_connectivity": seed_row[
                                "seed_inchi_key_connectivity"
                            ],
                            "seed_source": seed_row["seed_source"],
                            "mutation_depth": candidate.mutation_depth,
                            "mutation_operations": json.dumps(
                                candidate.mutation_operations, sort_keys=True
                            ),
                            "raw_selfies": candidate.raw_selfies,
                            "candidate_smiles": candidate.smiles,
                            "candidate_inchi_key_connectivity": candidate.inchi_key_connectivity,
                        }
                    )
                rejection_totals.update(
                    {f"{role}_{key}": value for key, value in stats.rejection_counts.items()}
                )
                generator_stats[role].update(
                    {
                        "proposal_budget": proposal_budget,
                        "proposals_considered": stats.proposals_considered,
                        "valid_molecules": stats.valid_molecules,
                        "exact_formula_molecules": stats.exact_formula_molecules,
                        "accepted_unique": stats.accepted_unique,
                    }
                )
            else:
                candidates, stats = generate_two_switch_neighbors(
                    seed_row["seed_smiles"],
                    seed=per_seed,
                    max_proposals=proposal_budget,
                    max_neighbors=accepted_budget,
                    exclude_connectivity_keys=original_keys[spec_name],
                    continue_after_neighbor_budget=True,
                )
                for candidate in candidates:
                    accepted_rows.append(
                        {
                            "query_spec_name": spec_name,
                            "generator": role,
                            "seed_smiles": seed_row["seed_smiles"],
                            "seed_inchi_key_connectivity": seed_row[
                                "seed_inchi_key_connectivity"
                            ],
                            "seed_source": seed_row["seed_source"],
                            "mutation_depth": 1,
                            "mutation_operations": json.dumps(
                                {
                                    "removed_bonds": candidate.removed_bonds,
                                    "added_bonds": candidate.added_bonds,
                                },
                                sort_keys=True,
                            ),
                            "raw_selfies": "",
                            "candidate_smiles": candidate.smiles,
                            "candidate_inchi_key_connectivity": candidate.inchi_key_connectivity,
                        }
                    )
                rejection_totals.update(
                    {f"{role}_{key}": value for key, value in stats.invalid_counts.items()}
                )
                generator_stats[role].update(
                    {
                        "proposal_budget": proposal_budget,
                        "proposals_available": stats.proposals_available,
                        "proposals_considered": stats.proposals_considered,
                        "accepted_unique": stats.accepted_unique,
                    }
                )
            timing_rows.append(
                {
                    "query_spec_name": spec_name,
                    "generator": role,
                    "wall_seconds": time.perf_counter() - started,
                }
            )

    accepted = pd.DataFrame(accepted_rows)
    if not accepted.empty:
        accepted = accepted.drop_duplicates(
            ["query_spec_name", "generator", "candidate_inchi_key_connectivity"],
            keep="first",
        )
    else:
        accepted = pd.DataFrame(
            columns=[
                "query_spec_name",
                "generator",
                "seed_smiles",
                "seed_inchi_key_connectivity",
                "seed_source",
                "mutation_depth",
                "mutation_operations",
                "raw_selfies",
                "candidate_smiles",
                "candidate_inchi_key_connectivity",
            ]
        )
    raw = pd.DataFrame(raw_rows)
    timing = pd.DataFrame(timing_rows)
    summary = {
        "rejection_counts": dict(sorted(rejection_totals.items())),
        "generators": {
            name: dict(sorted(counter.items()))
            for name, counter in sorted(generator_stats.items())
        },
    }
    stoned_summary = summary["generators"].get("stoned_C", {})
    stoned_proposals = max(int(stoned_summary.get("proposals_considered", 0)), 1)
    stoned_summary["validity_rate"] = (
        int(stoned_summary.get("valid_molecules", 0)) / stoned_proposals
    )
    stoned_summary["exact_formula_survival_rate"] = (
        int(stoned_summary.get("exact_formula_molecules", 0)) / stoned_proposals
    )
    return raw, accepted, summary, timing


def load_mist_fingerprints(
    metadata_path: Path, fingerprint_path: Path, spec_names: list[str]
) -> dict[str, np.ndarray]:
    columns = set(pd.read_csv(metadata_path, nrows=0).columns)
    expected_columns = {"fingerprint_index", "spec_name"}
    if columns != expected_columns:
        raise ValueError(
            "MIST ranking metadata must be sanitized to exactly "
            f"{sorted(expected_columns)}; got {sorted(columns)}"
        )
    metadata = pd.read_csv(
        metadata_path,
        usecols=["fingerprint_index", "spec_name"],
        dtype={"spec_name": str},
    )
    archive = np.load(fingerprint_path)
    if "mist_binary" not in archive:
        raise ValueError("Fingerprint archive is missing mist_binary")
    values = archive["mist_binary"]
    mapping: dict[str, np.ndarray] = {}
    for row in metadata.to_dict(orient="records"):
        mapping[str(row["spec_name"])] = values[int(row["fingerprint_index"])].astype(
            np.float32
        )
    missing = set(spec_names).difference(mapping)
    if missing:
        raise ValueError(f"Missing MIST fingerprints: {sorted(missing)}")
    return {name: mapping[name] for name in spec_names}


def tanimoto(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    intersection = float(np.minimum(fp_a, fp_b).sum())
    union = float(np.maximum(fp_a, fp_b).sum())
    return intersection / union if union else 0.0


def build_variant_pools(
    baseline: dict[str, dict[str, dict[str, Any]]],
    accepted: pd.DataFrame,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    additions: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in accepted.to_dict(orient="records"):
        record = _candidate_record(row["candidate_smiles"])
        if record is None:
            continue
        additions[row["query_spec_name"]][row["generator"]][
            record["candidate_inchi_key_connectivity"]
        ] = {**record, "generator": row["generator"]}

    output: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for spec_name, baseline_by_key in baseline.items():
        for variant, generators in VARIANTS.items():
            by_key = dict(baseline_by_key)
            for generator in generators:
                for key, record in additions[spec_name][generator].items():
                    by_key.setdefault(key, record)
            output[spec_name][variant] = list(by_key.values())
    return output


def rank_variant_pools(
    pools: dict[str, dict[str, list[dict[str, Any]]]],
    mist_fingerprints: dict[str, np.ndarray],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    ranked: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for spec_name, variants in pools.items():
        query_fp = mist_fingerprints[spec_name]
        for variant, candidates in variants.items():
            scored = [
                {**candidate, "mist_similarity": tanimoto(query_fp, candidate["fingerprint"])}
                for candidate in candidates
            ]
            scored.sort(
                key=lambda row: (
                    -row["mist_similarity"],
                    row.get("source_rank", 10**9),
                    row["candidate_smiles"],
                )
            )
            ranked[spec_name][variant] = scored
    return ranked


def ranked_candidates_frame(
    ranked: dict[str, dict[str, list[dict[str, Any]]]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec_name, variants in ranked.items():
        for variant, candidates in variants.items():
            for rank, candidate in enumerate(candidates, start=1):
                rows.append(
                    {
                        "query_spec_name": spec_name,
                        "variant": variant,
                        "rank": rank,
                        "candidate_smiles": candidate["candidate_smiles"],
                        "candidate_inchi_key_connectivity": candidate[
                            "candidate_inchi_key_connectivity"
                        ],
                        "candidate_formula": candidate["candidate_formula"],
                        "origin": candidate["generator"],
                        "mist_similarity": candidate["mist_similarity"],
                    }
                )
    frame = pd.DataFrame(rows)
    leaked = TARGET_COLUMNS.intersection(frame.columns)
    if leaked:
        raise AssertionError(f"Target fields leaked into ranking: {sorted(leaked)}")
    return frame


def load_targets(path: Path, spec_names: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    spec_col = "spec_name" if "spec_name" in frame else "query_spec_name"
    smiles_col = "target_smiles" if "target_smiles" in frame else "smiles"
    if spec_col not in frame or smiles_col not in frame:
        raise ValueError("Target CSV requires spec_name and target_smiles/smiles")
    frame = frame.set_index(spec_col, verify_integrity=True).loc[spec_names].reset_index()
    rows = []
    for row in frame.to_dict(orient="records"):
        record = _candidate_record(row[smiles_col])
        if record is None:
            raise ValueError(f"Invalid target SMILES for {row[spec_col]}")
        rows.append(
            {
                "spec_name": row[spec_col],
                "target_smiles": record["candidate_smiles"],
                "target_inchi_key_connectivity": record[
                    "candidate_inchi_key_connectivity"
                ],
                "target_fingerprint": record["fingerprint"],
            }
        )
    return pd.DataFrame(rows)


def evaluate_variants(
    ranked: dict[str, dict[str, list[dict[str, Any]]]], targets: pd.DataFrame
) -> pd.DataFrame:
    target_map = targets.set_index("spec_name", verify_integrity=True)
    rows: list[dict[str, Any]] = []
    for spec_name, variants in ranked.items():
        target = target_map.loc[spec_name]
        target_key = target["target_inchi_key_connectivity"]
        target_fp = target["target_fingerprint"]
        for variant, candidates in variants.items():
            keys = [candidate["candidate_inchi_key_connectivity"] for candidate in candidates]
            target_sims = [
                tanimoto(target_fp, candidate["fingerprint"])
                for candidate in candidates
            ]
            rows.append(
                {
                    "query_spec_name": spec_name,
                    "variant": variant,
                    "candidate_count": len(candidates),
                    "candidate_recall": int(target_key in keys),
                    "best_candidate_tanimoto": max(target_sims, default=0.0),
                    "mist_ranked_exact_top1": int(bool(keys) and keys[0] == target_key),
                    "mist_ranked_exact_top10": int(target_key in keys[:10]),
                    "mist_ranked_tanimoto_top1": target_sims[0] if target_sims else 0.0,
                    "mist_ranked_tanimoto_top10": max(target_sims[:10], default=0.0),
                }
            )
    return pd.DataFrame(rows)


def attach_variant_runtime(metrics: pd.DataFrame, timing: pd.DataFrame) -> pd.DataFrame:
    runtime = timing.groupby(["query_spec_name", "generator"])["wall_seconds"].sum()
    generators = {
        "A_union": (),
        "B_union_two_switch": ("two_switch_B",),
        "C_union_stoned": ("stoned_C",),
        "D_union_two_switch_stoned": ("two_switch_D", "stoned_D"),
    }
    output = metrics.copy()
    output["runtime_seconds"] = [
        sum(runtime.get((row.query_spec_name, name), 0.0) for name in generators[row.variant])
        for row in output.itertuples(index=False)
    ]
    return output


def bootstrap_mean_ci(
    values: np.ndarray, *, seed: int, samples: int = 10_000
) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def paired_comparisons(metrics: pd.DataFrame, seed: int) -> dict[str, Any]:
    metric_names = [
        "candidate_recall",
        "best_candidate_tanimoto",
        "mist_ranked_exact_top1",
        "mist_ranked_exact_top10",
        "mist_ranked_tanimoto_top1",
        "mist_ranked_tanimoto_top10",
        "runtime_seconds",
    ]
    baseline = metrics[metrics["variant"] == "A_union"].set_index(
        "query_spec_name", verify_integrity=True
    )
    output: dict[str, Any] = {}
    for variant in VARIANTS:
        if variant == "A_union":
            continue
        candidate = metrics[metrics["variant"] == variant].set_index(
            "query_spec_name", verify_integrity=True
        )
        output[variant] = {}
        for metric in metric_names:
            delta = (candidate[metric] - baseline[metric]).to_numpy(dtype=float)
            low, high = bootstrap_mean_ci(
                delta, seed=stable_seed(seed, variant, metric)
            )
            output[variant][metric] = {
                "baseline_mean": float(baseline[metric].mean()),
                "candidate_mean": float(candidate[metric].mean()),
                "mean_delta": float(delta.mean()),
                "ci95": [low, high],
            }
    return output


def summarize_ranking(metrics: pd.DataFrame) -> dict[str, Any]:
    numeric = [
        "candidate_count",
        "candidate_recall",
        "best_candidate_tanimoto",
        "mist_ranked_exact_top1",
        "mist_ranked_exact_top10",
        "mist_ranked_tanimoto_top1",
        "mist_ranked_tanimoto_top10",
        "runtime_seconds",
    ]
    return {
        variant: {
            metric: float(rows[metric].mean())
            for metric in numeric
        }
        for variant, rows in metrics.groupby("variant", sort=False)
    }


def decide(
    metrics: pd.DataFrame,
    comparisons: dict[str, Any],
    accepted: pd.DataFrame,
    rejection_statistics: dict[str, Any],
    timing: pd.DataFrame,
    dlm_runtime_seconds_per_query: float,
    panel_semantics: str,
) -> dict[str, Any]:
    baseline = metrics[metrics["variant"] == "A_union"].set_index("query_spec_name")
    stoned = metrics[metrics["variant"] == "C_union_stoned"].set_index(
        "query_spec_name"
    )
    new_recoveries = int(
        ((stoned["candidate_recall"] == 1) & (baseline["candidate_recall"] == 0)).sum()
    )
    best = comparisons["C_union_stoned"]["best_candidate_tanimoto"]
    stoned_stats = rejection_statistics["generators"].get("stoned_C", {})
    proposals = max(int(stoned_stats.get("proposals_considered", 0)), 1)
    exact_formula = int(stoned_stats.get("exact_formula_molecules", 0))
    survival = exact_formula / proposals
    stoned_keys = set(
        accepted.loc[accepted["generator"] == "stoned_C"]
        .loc[:, ["query_spec_name", "candidate_inchi_key_connectivity"]]
        .itertuples(index=False, name=None)
    )
    two_keys = set(
        accepted.loc[accepted["generator"] == "two_switch_B"]
        .loc[:, ["query_spec_name", "candidate_inchi_key_connectivity"]]
        .itertuples(index=False, name=None)
    )
    stoned_only = len(stoned_keys.difference(two_keys))
    stoned_runtime = timing[timing["generator"] == "stoned_C"].groupby(
        "query_spec_name"
    )["wall_seconds"].sum()
    runtime_per_query = float(stoned_runtime.mean()) if not stoned_runtime.empty else 0.0
    primary_pass = new_recoveries >= 1 or (
        best["mean_delta"] >= 0.01 and best["ci95"][0] > 0.0
    )
    stop_reasons = []
    if new_recoveries == 0 and best["mean_delta"] <= 0.0:
        stop_reasons.append("no_target_recovery_or_best_tanimoto_gain")
    if survival < 0.05:
        stop_reasons.append("formula_rejection_exceeds_95_percent")
    if stoned_only == 0:
        stop_reasons.append("no_connectivity_beyond_two_switch")
    if runtime_per_query >= dlm_runtime_seconds_per_query:
        stop_reasons.append("runtime_not_cheaper_than_dlm")
    exact10_delta = comparisons["C_union_stoned"]["mist_ranked_exact_top10"][
        "mean_delta"
    ]
    positive_subthreshold = best["mean_delta"] > 0.0 and best["ci95"][0] > 0.0
    if stop_reasons:
        decision = "rejected"
    elif panel_semantics == "target_absent_diagnostic" and primary_pass:
        decision = "bounded"
    elif primary_pass and exact10_delta > 0:
        decision = "promoted"
    elif primary_pass or positive_subthreshold:
        decision = "bounded"
    else:
        decision = "rejected"
    next_gate = "micro128" if primary_pass and decision in {"promoted", "bounded"} else None
    next_action = None
    if positive_subthreshold and not primary_pass:
        next_action = "run_predeclared_mutation_operator_ablation"
    return {
        "decision": decision,
        "new_target_recoveries": new_recoveries,
        "best_candidate_tanimoto_delta": best,
        "mist_ranked_exact_top10_mean_delta": exact10_delta,
        "exact_formula_survival_rate": survival,
        "stoned_unique_connectivity_not_in_two_switch": stoned_only,
        "stoned_runtime_seconds_per_query": runtime_per_query,
        "dlm_runtime_seconds_per_query_guard": dlm_runtime_seconds_per_query,
        "stop_reasons": stop_reasons,
        "gate_passed": primary_pass,
        "next_gate": next_gate,
        "next_action": next_action,
        "panel_semantics": panel_semantics,
        "can_promote_from_this_panel": panel_semantics == "target_blind",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Target-blind constrained STONED-SELFIES candidate expansion."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fixed-spec-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--queries-csv", type=Path, required=True)
    parser.add_argument("--union-csv", type=Path, required=True)
    parser.add_argument("--molforge-csv", type=Path, required=True)
    parser.add_argument("--retrieval-csv", type=Path)
    parser.add_argument("--mist-metadata-csv", type=Path, required=True)
    parser.add_argument("--mist-fingerprints-npz", type=Path, required=True)
    parser.add_argument("--evaluation-targets-csv", type=Path, required=True)
    parser.add_argument("--max-seeds", type=int, default=8)
    parser.add_argument("--seeds-per-source", type=int, default=4)
    parser.add_argument("--raw-proposals-per-seed", type=int, default=512)
    parser.add_argument("--accepted-per-seed", type=int, default=64)
    parser.add_argument("--mutation-depths", default="1,2")
    parser.add_argument("--operations", default="replacement")
    parser.add_argument("--seed", type=int, default=13420260711)
    parser.add_argument("--dlm-runtime-seconds-per-query", type=float, default=42.0)
    parser.add_argument("--expected-host", default="spectrum")
    parser.add_argument(
        "--panel-semantics",
        choices=("target_blind", "target_absent_diagnostic"),
        required=True,
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    actual_host = socket.gethostname().split(".", maxsplit=1)[0].lower()
    if actual_host != args.expected_host.lower():
        raise RuntimeError(
            f"STONED production runs are restricted to {args.expected_host}; got {actual_host}"
        )
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    args.run_dir.mkdir(parents=True)
    log_path = args.run_dir / "stdout.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        tee = Tee(sys.__stdout__, log_handle)
        with redirect_stdout(tee), redirect_stderr(tee):
            print("Starting constrained STONED-SELFIES expansion")
            spec_names, manifest_hash = load_fixed_manifest(
                args.fixed_spec_manifest, args.expected_manifest_sha256
            )
            queries = load_queries(args.queries_csv, spec_names)
            query_formulas = dict(zip(queries["spec_name"], queries["formula"]))
            union = load_candidates(args.union_csv, "union")
            molforge = load_candidates(args.molforge_csv, "molforge")
            source_frames = [("union", union), ("molforge", molforge)]
            input_paths = {
                "fixed_spec_manifest": args.fixed_spec_manifest,
                "queries_csv": args.queries_csv,
                "union_csv": args.union_csv,
                "molforge_csv": args.molforge_csv,
                "mist_metadata_csv": args.mist_metadata_csv,
                "mist_fingerprints_npz": args.mist_fingerprints_npz,
            }
            if args.retrieval_csv:
                retrieval = load_candidates(args.retrieval_csv, "retrieval")
                source_frames.append(("retrieval", retrieval))
                input_paths["retrieval_csv"] = args.retrieval_csv

            assert_tracked_worktree_clean(PROJECT_ROOT)
            code_paths = {
                "runner": Path(__file__).resolve(),
                "stoned_expansion": SRC_ROOT / "frigid" / "stoned_expansion.py",
                "two_switch": SRC_ROOT / "frigid" / "two_switch.py",
            }
            mutation_depths = tuple(
                int(value) for value in args.mutation_depths.split(",")
            )
            operations = tuple(
                value.strip() for value in args.operations.split(",") if value.strip()
            )
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "status": "running",
                "host": actual_host,
                "platform": platform.platform(),
                "commit": git_commit(PROJECT_ROOT),
                "code_hashes": {
                    name: {"path": str(path), "sha256": sha256_file(path)}
                    for name, path in code_paths.items()
                },
                "selfies_version": sf.__version__,
                "input_hashes": {
                    name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                    for name, path in input_paths.items()
                },
                "fixed_manifest_sha256": manifest_hash,
                "panel_semantics": args.panel_semantics,
                "can_promote_from_this_panel": args.panel_semantics == "target_blind",
                "random_seed": args.seed,
                "mutation_settings": {
                    "depths": mutation_depths,
                    "operations": operations,
                },
                "budgets": {
                    "max_seeds": args.max_seeds,
                    "seeds_per_source": args.seeds_per_source,
                    "B_two_switch_proposals_per_seed": args.raw_proposals_per_seed,
                    "C_stoned_proposals_per_seed": args.raw_proposals_per_seed,
                    "D_two_switch_proposals_per_seed": args.raw_proposals_per_seed // 2,
                    "D_stoned_proposals_per_seed": args.raw_proposals_per_seed // 2,
                    "accepted_per_seed_single_source": args.accepted_per_seed,
                    "accepted_per_seed_each_combined_source": args.accepted_per_seed // 2,
                },
                "command": [sys.executable, *sys.argv],
                "start_timestamp": started_at.isoformat(),
                "end_timestamp": None,
                "query_count": len(spec_names),
                "target_use": {
                    "selection": (
                        "predeclared_target_absent_panel"
                        if args.panel_semantics == "target_absent_diagnostic"
                        else []
                    ),
                    "generation": [],
                    "filtering": [],
                    "budgeting": [],
                    "ranking": [],
                    "metrics_only": sorted(TARGET_COLUMNS),
                },
                "decision": None,
            }
            _write_json(args.run_dir / "RUN_MANIFEST.json", manifest)

            baseline = build_baseline_pool(union, spec_names)
            seeds, original_keys, seed_rejections = select_seeds(
                spec_names,
                query_formulas,
                source_frames,
                seeds_per_source=args.seeds_per_source,
                max_seeds=args.max_seeds,
            )
            raw, accepted, rejection_statistics, timing = generate_expansions(
                seeds,
                original_keys,
                global_seed=args.seed,
                raw_proposals_per_seed=args.raw_proposals_per_seed,
                accepted_per_seed=args.accepted_per_seed,
                mutation_depths=mutation_depths,
                operations=operations,
            )
            rejection_statistics["seed_rejections"] = dict(sorted(seed_rejections.items()))
            queries.to_csv(args.run_dir / "queries.csv", index=False)
            seeds.to_csv(args.run_dir / "seeds.csv", index=False)
            raw.to_csv(args.run_dir / "raw_proposals.csv", index=False)
            accepted.to_csv(args.run_dir / "accepted_candidates.csv", index=False)
            _write_json(
                args.run_dir / "rejection_statistics.json", rejection_statistics
            )

            forbidden_generation = TARGET_COLUMNS.intersection(queries.columns)
            forbidden_generation.update(TARGET_COLUMNS.intersection(seeds.columns))
            forbidden_generation.update(TARGET_COLUMNS.intersection(raw.columns))
            forbidden_generation.update(TARGET_COLUMNS.intersection(accepted.columns))
            if forbidden_generation:
                raise AssertionError(
                    f"Target fields leaked into generation artifacts: {sorted(forbidden_generation)}"
                )

            mist = load_mist_fingerprints(
                args.mist_metadata_csv, args.mist_fingerprints_npz, spec_names
            )
            pools = build_variant_pools(baseline, accepted)
            ranked = rank_variant_pools(pools, mist)
            ranked_frame = ranked_candidates_frame(ranked)
            ranked_frame.to_csv(args.run_dir / "ranked_candidates.csv", index=False)
            targets = load_targets(args.evaluation_targets_csv, spec_names)
            manifest["input_hashes"]["evaluation_targets_csv"] = {
                "path": str(args.evaluation_targets_csv.resolve()),
                "sha256": sha256_file(args.evaluation_targets_csv),
            }
            metrics = evaluate_variants(ranked, targets)
            metrics = attach_variant_runtime(metrics, timing)
            comparisons = paired_comparisons(metrics, args.seed)
            ranking_summary = summarize_ranking(metrics)
            decision = decide(
                metrics,
                comparisons,
                accepted,
                rejection_statistics,
                timing,
                args.dlm_runtime_seconds_per_query,
                args.panel_semantics,
            )
            metrics.to_csv(args.run_dir / "candidate_recall.csv", index=False)
            _write_json(
                args.run_dir / "ranking_metrics.json",
                {
                    "variants": ranking_summary,
                    "target_fields_used_by_generation": [],
                    "target_fields_used_by_ranking": [],
                    "target_fields_used_for_metrics_only": sorted(TARGET_COLUMNS),
                },
            )
            _write_json(
                args.run_dir / "paired_comparison.json",
                {"comparisons": comparisons, "decision": decision},
            )

            manifest["status"] = "completed"
            manifest["end_timestamp"] = datetime.now(timezone.utc).isoformat()
            manifest["decision"] = decision
            manifest["output_hashes"] = {
                path.name: sha256_file(path)
                for path in sorted(args.run_dir.iterdir())
                if path.is_file() and path.name not in {"RUN_MANIFEST.json", "EXIT_STATUS"}
            }
            _write_json(args.run_dir / "RUN_MANIFEST.json", manifest)
            print(json.dumps(decision, indent=2, sort_keys=True))
            return manifest


def main() -> int:
    args = parse_args()
    status = 1
    try:
        run(args)
        status = 0
        return 0
    except Exception as exc:
        manifest_path = args.run_dir / "RUN_MANIFEST.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            manifest["status"] = "failed"
            manifest["end_timestamp"] = datetime.now(timezone.utc).isoformat()
            manifest["error"] = f"{type(exc).__name__}: {exc}"
            _write_json(manifest_path, manifest)
        raise
    finally:
        if args.run_dir.exists():
            (args.run_dir / "EXIT_STATUS").write_text(f"{status}\n")


if __name__ == "__main__":
    raise SystemExit(main())
