#!/usr/bin/env python
"""Run a bounded, target-blind GEMS-style ICEBERG diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from frigid.two_switch import generate_two_switch_neighbors  # noqa: E402


SCHEMA_VERSION = 1
TARGET_COLUMNS = {"target_smiles", "target_inchi_key", "target_inchi_key_connectivity"}
DEFAULT_MS_PRED_COMMIT = "00948ecfc171b3480e549c31bf765197fa0eb30d"
DEFAULT_COLLISION_ENERGIES = (10, 20, 30, 40, 50)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def connectivity_key(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    return key.split("-", maxsplit=1)[0] if key else None


def molecular_formula(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return None


def normalize_instrument(value: object) -> str:
    instrument = str(value).strip().lower()
    if "orbitrap" in instrument:
        return "Orbitrap"
    if "qtof" in instrument or "q-tof" in instrument:
        return "QTOF"
    raise ValueError(f"Unsupported ICEBERG instrument: {value!r}")


def load_union_candidates(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"query_spec_name", "rank", "candidate_smiles"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"Candidate table {path} is missing columns: {sorted(missing)}"
        )
    frame = frame.copy()
    frame["query_spec_name"] = frame["query_spec_name"].astype(str)
    frame["rank"] = pd.to_numeric(frame["rank"], errors="raise").astype(int)
    frame["candidate_smiles"] = frame["candidate_smiles"].astype(str)
    return frame.sort_values(["query_spec_name", "rank"], kind="stable")


def _top_candidate_keys(frame: pd.DataFrame, spec_name: str, top_k: int) -> set[str]:
    rows = frame[frame["query_spec_name"] == spec_name].nsmallest(top_k, "rank")
    return {
        key
        for key in (connectivity_key(smiles) for smiles in rows["candidate_smiles"])
        if key is not None
    }


def select_hard_queries(
    metadata: pd.DataFrame,
    current: pd.DataFrame,
    molforge: pd.DataFrame,
    *,
    limit: int,
    selection_seed: int,
    top_k: int = 10,
) -> tuple[list[str], pd.DataFrame, int]:
    """Select target-absent queries, then order them without target information."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    metadata = metadata.copy()
    if "spec_name" not in metadata or "smiles" not in metadata:
        raise ValueError("Metadata must contain spec_name and smiles")
    metadata["spec_name"] = metadata["spec_name"].astype(str)
    if metadata["spec_name"].duplicated().any():
        raise ValueError("Metadata contains duplicate spec_name values")

    eligible: list[dict[str, str]] = []
    for row in metadata.to_dict(orient="records"):
        spec_name = str(row["spec_name"])
        target_smiles = str(row["smiles"])
        target_key = str(row.get("inchi_key_first_block", "")).strip()
        if not target_key or target_key.lower() == "nan":
            target_key = connectivity_key(target_smiles) or ""
        if not target_key:
            raise ValueError(f"Cannot derive target connectivity key for {spec_name}")
        unchanged_keys = _top_candidate_keys(current, spec_name, top_k)
        unchanged_keys.update(_top_candidate_keys(molforge, spec_name, top_k))
        if target_key in unchanged_keys:
            continue
        eligible.append(
            {
                "spec_name": spec_name,
                "target_smiles": target_smiles,
                "target_inchi_key_connectivity": target_key,
            }
        )

    eligible.sort(
        key=lambda row: (
            stable_seed(selection_seed, row["spec_name"]),
            row["spec_name"],
        )
    )
    selected = eligible[:limit]
    if len(selected) < limit:
        raise ValueError(
            f"Only {len(selected)} hard queries are available; requested {limit}"
        )
    targets = pd.DataFrame(selected)
    return targets["spec_name"].tolist(), targets, len(eligible)


def _candidate_records(
    frame: pd.DataFrame,
    spec_name: str,
    origin: str,
    top_k: int,
) -> list[dict[str, Any]]:
    rows = frame[frame["query_spec_name"] == spec_name].nsmallest(top_k, "rank")
    return [
        {
            "origin": origin,
            "source_rank": int(row["rank"]),
            "candidate_smiles": str(row["candidate_smiles"]),
        }
        for row in rows.to_dict(orient="records")
    ]


def _round_robin_seeds(
    current_rows: list[dict[str, Any]],
    molforge_rows: list[dict[str, Any]],
    *,
    seeds_per_source: int,
    max_seeds: int,
    query_formula: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    rejected: Counter[str] = Counter()
    source_rows = [
        current_rows[:seeds_per_source],
        molforge_rows[:seeds_per_source],
    ]
    for rank_index in range(seeds_per_source):
        for rows in source_rows:
            if rank_index >= len(rows) or len(selected) >= max_seeds:
                continue
            row = rows[rank_index]
            key = connectivity_key(row["candidate_smiles"])
            formula = molecular_formula(row["candidate_smiles"])
            if key is None or formula is None:
                rejected["invalid_seed"] += 1
                continue
            if formula != query_formula:
                rejected["formula_mismatch_seed"] += 1
                continue
            if key in seen:
                rejected["duplicate_seed"] += 1
                continue
            seen.add(key)
            selected.append({**row, "candidate_inchi_key_connectivity": key})
    return selected, rejected


def build_search_space(
    selected_spec_names: list[str],
    current: pd.DataFrame,
    molforge: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    spec_dir: Path,
    baseline_top_k: int,
    seeds_per_source: int,
    max_seeds: int,
    neighbor_seed: int,
    max_proposals_per_seed: int,
    max_neighbors_per_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build candidates without accepting target structures or fingerprints."""

    if TARGET_COLUMNS.intersection(labels.columns):
        raise ValueError("Target columns are forbidden in search labels")
    required_labels = {"spec", "formula", "ionization", "instrument"}
    missing = required_labels.difference(labels.columns)
    if missing:
        raise ValueError(f"Labels are missing columns: {sorted(missing)}")
    label_map = labels.set_index("spec", verify_integrity=True)

    query_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    edit_rows: list[dict[str, Any]] = []

    for spec_name in selected_spec_names:
        if spec_name not in label_map.index:
            raise ValueError(f"Missing label for {spec_name}")
        label = label_map.loc[spec_name]
        query_formula = str(label["formula"]).strip()
        ionization = str(label["ionization"]).strip()
        instrument = normalize_instrument(label["instrument"])
        observed_path = spec_dir / f"{spec_name}.ms"
        if not observed_path.is_file():
            raise FileNotFoundError(observed_path)
        query_rows.append(
            {
                "spec_name": spec_name,
                "formula": query_formula,
                "ionization": ionization,
                "instrument": instrument,
                "observed_spectrum_path": str(observed_path.resolve()),
            }
        )

        current_rows = _candidate_records(
            current, spec_name, "current_union", baseline_top_k
        )
        molforge_rows = _candidate_records(
            molforge, spec_name, "molforge_union", baseline_top_k
        )
        seeds, seed_rejections = _round_robin_seeds(
            current_rows,
            molforge_rows,
            seeds_per_source=seeds_per_source,
            max_seeds=max_seeds,
            query_formula=query_formula,
        )
        seed_keys = {row["candidate_inchi_key_connectivity"] for row in seeds}

        by_key: dict[str, dict[str, Any]] = {}
        baseline_rejections: Counter[str] = Counter()
        for row in [*current_rows, *molforge_rows]:
            key = connectivity_key(row["candidate_smiles"])
            formula = molecular_formula(row["candidate_smiles"])
            if key is None or formula is None:
                baseline_rejections["invalid_baseline"] += 1
                continue
            if formula != query_formula:
                baseline_rejections["formula_mismatch_baseline"] += 1
                continue
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = {
                    "query_spec_name": spec_name,
                    "candidate_smiles": row["candidate_smiles"],
                    "candidate_inchi_key_connectivity": key,
                    "candidate_formula": formula,
                    "origin": row["origin"],
                    "source_rank": row["source_rank"],
                    "is_seed": int(key in seed_keys),
                    "is_neighbor": 0,
                    "seed_smiles": "",
                    "seed_inchi_key_connectivity": "",
                    "removed_bonds": "",
                    "added_bonds": "",
                }
            elif row["origin"] not in existing["origin"].split("+"):
                existing["origin"] += f"+{row['origin']}"

        excluded_keys = set(by_key)
        for seed_index, seed_row in enumerate(seeds):
            per_seed = stable_seed(
                neighbor_seed,
                spec_name,
                seed_row["candidate_inchi_key_connectivity"],
            )
            started = time.perf_counter()
            neighbors, stats = generate_two_switch_neighbors(
                seed_row["candidate_smiles"],
                seed=per_seed,
                max_proposals=max_proposals_per_seed,
                max_neighbors=max_neighbors_per_seed,
                exclude_connectivity_keys=excluded_keys,
            )
            elapsed = time.perf_counter() - started
            for neighbor in neighbors:
                excluded_keys.add(neighbor.inchi_key_connectivity)
                by_key[neighbor.inchi_key_connectivity] = {
                    "query_spec_name": spec_name,
                    "candidate_smiles": neighbor.smiles,
                    "candidate_inchi_key_connectivity": neighbor.inchi_key_connectivity,
                    "candidate_formula": query_formula,
                    "origin": "two_switch",
                    "source_rank": seed_index + 1,
                    "is_seed": 0,
                    "is_neighbor": 1,
                    "seed_smiles": seed_row["candidate_smiles"],
                    "seed_inchi_key_connectivity": seed_row[
                        "candidate_inchi_key_connectivity"
                    ],
                    "removed_bonds": json.dumps(neighbor.removed_bonds),
                    "added_bonds": json.dumps(neighbor.added_bonds),
                }
            edit_rows.append(
                {
                    "query_spec_name": spec_name,
                    "seed_index": seed_index,
                    "seed_smiles": seed_row["candidate_smiles"],
                    "seed_inchi_key_connectivity": seed_row[
                        "candidate_inchi_key_connectivity"
                    ],
                    "seed_origin": seed_row["origin"],
                    "rng_seed": per_seed,
                    "proposals_available": stats.proposals_available,
                    "proposals_considered": stats.proposals_considered,
                    "accepted_unique": stats.accepted_unique,
                    "invalid_counts_json": json.dumps(
                        dict(sorted(stats.invalid_counts.items())), sort_keys=True
                    ),
                    "generation_wall_seconds": elapsed,
                    "seed_rejections_json": json.dumps(
                        dict(sorted(seed_rejections.items())), sort_keys=True
                    ),
                    "baseline_rejections_json": json.dumps(
                        dict(sorted(baseline_rejections.items())), sort_keys=True
                    ),
                }
            )

        for candidate_index, row in enumerate(by_key.values()):
            candidate_rows.append({"candidate_index": candidate_index, **row})

    queries = pd.DataFrame(query_rows)
    candidates = pd.DataFrame(candidate_rows)
    edits = pd.DataFrame(edit_rows)
    forbidden = TARGET_COLUMNS.intersection(queries.columns).union(
        TARGET_COLUMNS.intersection(candidates.columns)
    )
    if forbidden:
        raise AssertionError(
            f"Target fields leaked into search artifacts: {sorted(forbidden)}"
        )
    return queries, candidates, edits


def write_prepare_artifacts(
    output_dir: Path,
    queries: pd.DataFrame,
    candidates: pd.DataFrame,
    edits: pd.DataFrame,
    targets: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {
        "search_queries_csv": output_dir / "search_queries.csv",
        "candidates_csv": output_dir / "candidates.csv",
        "edit_statistics_csv": output_dir / "edit_statistics.csv",
        "evaluation_targets_csv": output_dir / "evaluation_targets.csv",
    }
    queries.to_csv(paths["search_queries_csv"], index=False)
    candidates.to_csv(paths["candidates_csv"], index=False)
    edits.to_csv(paths["edit_statistics_csv"], index=False)
    targets.to_csv(paths["evaluation_targets_csv"], index=False)
    manifest["outputs"] = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    (output_dir / "prepare_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def verify_official_ms_pred(ms_pred_root: Path, expected_commit: str) -> dict[str, str]:
    commit = _git_output(ms_pred_root, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise RuntimeError(f"ms-pred commit mismatch: {commit} != {expected_commit}")
    status = _git_output(ms_pred_root, "status", "--porcelain=v1")
    if status:
        raise RuntimeError(f"ms-pred checkout is dirty:\n{status}")
    origin = _git_output(ms_pred_root, "remote", "get-url", "origin")
    if origin.rstrip("/") != "https://github.com/coleygroup/ms-pred.git":
        raise RuntimeError(f"Unexpected ms-pred origin: {origin}")
    license_path = ms_pred_root / "LICENSE"
    return {
        "commit": commit,
        "origin": origin,
        "license": "MIT",
        "license_sha256": sha256_file(license_path),
    }


def _run_official_iceberg(
    *,
    query: dict[str, Any],
    candidates: pd.DataFrame,
    query_dir: Path,
    ms_pred_root: Path,
    python_path: Path,
    gen_checkpoint: Path,
    inten_checkpoint: Path,
    collision_energies: tuple[int, ...],
    gpu: int,
    batch_size: int,
    num_workers: int,
) -> tuple[Path, float]:
    sys.path.insert(0, str(ms_pred_root / "src"))
    import ms_pred.common as common

    query_dir.mkdir(parents=True, exist_ok=False)
    labels_path = query_dir / "iceberg_candidates.tsv"
    prediction_dir = query_dir / "prediction"
    prediction_dir.mkdir()
    entries = []
    for row in candidates.to_dict(orient="records"):
        smiles = str(row["candidate_smiles"])
        precursor = common.mass_from_smi(smiles) + common.ion2mass[query["ionization"]]
        entries.append(
            {
                "spec": query["spec_name"],
                "smiles": smiles,
                "ionization": query["ionization"],
                "instrument": query["instrument"],
                "inchikey": common.inchikey_from_smiles(smiles),
                "precursor": precursor,
                "collision_energies": [str(value) for value in collision_energies],
            }
        )
    pd.DataFrame(entries).to_csv(labels_path, sep="\t", index=False)

    official_script = ms_pred_root / "src/ms_pred/dag_pred/predict_smis.py"
    compat_runner = (
        "import runpy,sys; import pytorch_lightning as pl; "
        "pl.utilities.seed.seed_everything=pl.seed_everything; "
        "script=sys.argv[1]; sys.argv=[script,*sys.argv[2:]]; "
        "runpy.run_path(script,run_name='__main__')"
    )
    command = [
        str(python_path),
        "-c",
        compat_runner,
        str(official_script),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--dataset-labels",
        str(labels_path),
        "--sparse-out",
        "--sparse-k",
        "100",
        "--max-nodes",
        "100",
        "--threshold",
        "0.0",
        "--gen-checkpoint",
        str(gen_checkpoint),
        "--inten-checkpoint",
        str(inten_checkpoint),
        "--save-dir",
        str(prediction_dir),
        "--adduct-shift",
        "--seed",
        "42",
        "--gpu",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    pythonpath = [str(ms_pred_root / "src"), str(SRC_ROOT)]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    log_path = query_dir / "iceberg.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        result = subprocess.run(
            command,
            cwd=ms_pred_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
        raise RuntimeError(
            f"Official ICEBERG failed for {query['spec_name']} with exit "
            f"{result.returncode}:\n{tail}"
        )
    prediction_path = prediction_dir / "preds.hdf5"
    if not prediction_path.is_file():
        raise RuntimeError(f"ICEBERG produced no predictions: {prediction_path}")
    return prediction_path, elapsed


def sparse_cosine_similarity_20ppm(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    precursor_mz: float,
    ppm: int = 20,
    ignore_precursor: bool = True,
) -> float:
    """Match sparse peaks with the same cosine/Hungarian contract as ms-pred."""

    predicted = np.asarray(predicted, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if ignore_precursor:
        predicted = predicted[predicted[:, 0] <= precursor_mz - 1]
        observed = observed[observed[:, 0] <= precursor_mz - 1]
    if len(predicted) == 0 or len(observed) == 0:
        return 0.0
    pred_norm = np.linalg.norm(predicted[:, 1]) + 1e-22
    obs_norm = np.linalg.norm(observed[:, 1]) + 1e-22
    tolerance = precursor_mz * ppm * 1e-6
    compatible = np.abs(predicted[:, None, 0] - observed[None, :, 0]) < tolerance
    scores = predicted[:, None, 1] * observed[None, :, 1] / (pred_norm * obs_norm)
    scores *= compatible
    return _maximum_assignment_score(scores)


def _maximum_assignment_score(scores: np.ndarray) -> float:
    """Return a maximum-weight rectangular assignment in O(n^3)."""

    weights = np.asarray(scores, dtype=float)
    if weights.size == 0:
        return 0.0
    if weights.shape[0] > weights.shape[1]:
        weights = weights.T
    row_count, column_count = weights.shape
    max_weight = float(weights.max(initial=0.0))
    costs = max_weight - weights

    potentials_rows = np.zeros(row_count + 1)
    potentials_columns = np.zeros(column_count + 1)
    matching = np.zeros(column_count + 1, dtype=int)
    previous_column = np.zeros(column_count + 1, dtype=int)

    for row in range(1, row_count + 1):
        matching[0] = row
        current_column = 0
        min_costs = np.full(column_count + 1, np.inf)
        used = np.zeros(column_count + 1, dtype=bool)
        while True:
            used[current_column] = True
            current_row = matching[current_column]
            delta = np.inf
            next_column = 0
            for column in range(1, column_count + 1):
                if used[column]:
                    continue
                reduced = (
                    costs[current_row - 1, column - 1]
                    - potentials_rows[current_row]
                    - potentials_columns[column]
                )
                if reduced < min_costs[column]:
                    min_costs[column] = reduced
                    previous_column[column] = current_column
                if min_costs[column] < delta:
                    delta = min_costs[column]
                    next_column = column
            for column in range(column_count + 1):
                if used[column]:
                    potentials_rows[matching[column]] += delta
                    potentials_columns[column] -= delta
                else:
                    min_costs[column] -= delta
            current_column = next_column
            if matching[current_column] == 0:
                break
        while True:
            prior = previous_column[current_column]
            matching[current_column] = matching[prior]
            current_column = prior
            if current_column == 0:
                break

    assigned_columns = np.flatnonzero(matching[1:]) + 1
    assigned_rows = matching[assigned_columns]
    return float(weights[assigned_rows - 1, assigned_columns - 1].sum())


def _load_and_score_predictions(
    *,
    prediction_path: Path,
    candidates: pd.DataFrame,
    observed_path: Path,
    ms_pred_root: Path,
) -> tuple[list[dict[str, Any]], int]:
    sys.path.insert(0, str(ms_pred_root / "src"))
    import ms_pred.common as common
    from ms_pred.dag_pred.iceberg_elucidation import load_pred_spec, load_real_spec

    metadata, _ = common.parse_spectra(observed_path)
    precursor_mz = float(metadata["parentmass"])
    observed_by_key = load_real_spec(
        str(observed_path),
        "ms",
        precursor_mass=None,
        nce=False,
        ppm=20,
        denoise_spectrum=True,
    )
    observed_arrays = [
        np.asarray(value, dtype=float) for value in observed_by_key.values()
    ]
    observed = np.vstack(observed_arrays)
    pred_smiles, pred_specs, _ = load_pred_spec(str(prediction_path), merge_spec=False)
    predictions_by_key: dict[str, dict[str, np.ndarray]] = {}
    for smiles, spectra in zip(pred_smiles, pred_specs, strict=True):
        key = connectivity_key(str(smiles))
        if key is not None:
            predictions_by_key[key] = spectra

    rows: list[dict[str, Any]] = []
    missing = 0
    for candidate in candidates.to_dict(orient="records"):
        key = candidate["candidate_inchi_key_connectivity"]
        spectra = predictions_by_key.get(key)
        if spectra is None:
            missing += 1
            score = math.nan
            best_ce = math.nan
        else:
            by_energy = []
            for collision_energy, predicted in spectra.items():
                similarity = sparse_cosine_similarity_20ppm(
                    predicted,
                    observed,
                    precursor_mz=precursor_mz,
                    ppm=20,
                    ignore_precursor=True,
                )
                by_energy.append((similarity, float(collision_energy)))
            score, best_ce = max(by_energy, key=lambda item: (item[0], -item[1]))
        rows.append(
            {
                **candidate,
                "iceberg_score": score,
                "best_collision_energy_ev": best_ce,
                "precursor_mz": precursor_mz,
            }
        )
    return rows, missing


def morgan_tanimoto(smiles_a: str, smiles_b: str) -> float:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0
    fp_a = generator.GetFingerprint(mol_a)
    fp_b = generator.GetFingerprint(mol_b)
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def evaluate_scores(
    scores: pd.DataFrame,
    targets: pd.DataFrame,
    edits: pd.DataFrame,
    score_manifest: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_map = targets.set_index("spec_name", verify_integrity=True)
    details: list[dict[str, Any]] = []
    for spec_name, query_scores in scores.groupby("query_spec_name", sort=False):
        target = target_map.loc[spec_name]
        target_key = target["target_inchi_key_connectivity"]
        target_smiles = target["target_smiles"]
        before = query_scores[query_scores["is_neighbor"] == 0]
        recovered_before = target_key in set(before["candidate_inchi_key_connectivity"])
        recovered_after = target_key in set(
            query_scores["candidate_inchi_key_connectivity"]
        )

        ranked = query_scores.copy()
        ranked["finite_score"] = np.isfinite(ranked["iceberg_score"])
        ranked = ranked.sort_values(
            ["finite_score", "iceberg_score", "candidate_index"],
            ascending=[False, False, True],
            kind="stable",
        )
        top10 = ranked.head(10)
        exact_at10 = target_key in set(top10["candidate_inchi_key_connectivity"])
        tanimoto_at10 = max(
            (
                morgan_tanimoto(smiles, target_smiles)
                for smiles in top10["candidate_smiles"]
            ),
            default=0.0,
        )
        finite_scores = ranked.loc[ranked["finite_score"], "iceberg_score"].to_numpy()
        unique_scores = len(np.unique(np.round(finite_scores, 12)))
        score_range = (
            float(finite_scores.max() - finite_scores.min())
            if len(finite_scores)
            else 0.0
        )
        nondegenerate = unique_scores >= 2 and score_range > 1e-6
        details.append(
            {
                "spec_name": spec_name,
                "candidate_recovery_before": int(recovered_before),
                "candidate_recovery_after": int(recovered_after),
                "new_candidate_recovery": int(recovered_after and not recovered_before),
                "exact_match_top10": int(exact_at10),
                "tanimoto_top10": tanimoto_at10,
                "candidate_count": len(ranked),
                "finite_score_count": int(ranked["finite_score"].sum()),
                "unique_iceberg_scores": unique_scores,
                "iceberg_score_range": score_range,
                "forward_ranking_nondegenerate": int(nondegenerate),
            }
        )

    details_frame = pd.DataFrame(details)
    invalid_counts: Counter[str] = Counter()
    for payload in edits.get("invalid_counts_json", pd.Series(dtype=str)).dropna():
        invalid_counts.update(json.loads(payload))
    query_count = len(details_frame)
    recovered_before = int(details_frame["candidate_recovery_before"].sum())
    recovered_after = int(details_frame["candidate_recovery_after"].sum())
    new_recovery = int(details_frame["new_candidate_recovery"].sum())
    nondegenerate_fraction = float(
        details_frame["forward_ranking_nondegenerate"].mean()
    )
    gate_nondegenerate = nondegenerate_fraction >= 0.75
    if query_count == 4:
        decision = (
            "EXPAND_TO_16"
            if new_recovery > 0 and gate_nondegenerate
            else "REJECT_PILOT"
        )
    elif query_count >= 16:
        decision = (
            "PROMOTE_TO_64"
            if new_recovery > 0 and gate_nondegenerate
            else "REJECT_DIAGNOSTIC"
        )
    else:
        decision = "DEFER_INCOMPLETE_GATE"
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "query_count": query_count,
        "candidate_recovery_before": recovered_before,
        "candidate_recovery_after": recovered_after,
        "new_candidate_recoveries": new_recovery,
        "exact_match_top10": float(details_frame["exact_match_top10"].mean()),
        "tanimoto_top10_mean": float(details_frame["tanimoto_top10"].mean()),
        "forward_ranking_nondegenerate_fraction": nondegenerate_fraction,
        "forward_ranking_nondegenerate_gate": gate_nondegenerate,
        "valid_unique_edits": int(edits["accepted_unique"].sum()),
        "proposals_considered": int(edits["proposals_considered"].sum()),
        "invalid_edit_counts": dict(sorted(invalid_counts.items())),
        "unique_candidates": int(details_frame["candidate_count"].sum()),
        "iceberg_calls": score_manifest["iceberg_calls"],
        "iceberg_wall_seconds": score_manifest["iceberg_wall_seconds"],
        "decision": decision,
        "target_fields_used_by_generation_or_scoring": [],
    }
    return details_frame, aggregate


def command_prepare(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    current_path = Path(args.current_union).expanduser().resolve()
    molforge_path = Path(args.molforge_union).expanduser().resolve()
    metadata_path = Path(args.metadata_csv).expanduser().resolve()
    labels_path = Path(args.labels_tsv).expanduser().resolve()
    spec_dir = Path(args.spec_dir).expanduser().resolve()
    current = load_union_candidates(current_path)
    molforge = load_union_candidates(molforge_path)
    metadata = pd.read_csv(metadata_path)
    labels = pd.read_csv(labels_path, sep="\t")
    search_labels = labels[["spec", "formula", "ionization", "instrument"]].copy()
    selected, targets, eligible_count = select_hard_queries(
        metadata,
        current,
        molforge,
        limit=args.limit,
        selection_seed=args.selection_seed,
        top_k=args.baseline_top_k,
    )
    queries, candidates, edits = build_search_space(
        selected,
        current,
        molforge,
        search_labels,
        spec_dir=spec_dir,
        baseline_top_k=args.baseline_top_k,
        seeds_per_source=args.seeds_per_source,
        max_seeds=args.max_seeds,
        neighbor_seed=args.neighbor_seed,
        max_proposals_per_seed=args.max_proposals_per_seed,
        max_neighbors_per_seed=args.max_neighbors_per_seed,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "prepare",
        "inputs": {
            "current_union": {
                "path": str(current_path),
                "sha256": sha256_file(current_path),
            },
            "molforge_union": {
                "path": str(molforge_path),
                "sha256": sha256_file(molforge_path),
            },
            "metadata_csv": {
                "path": str(metadata_path),
                "sha256": sha256_file(metadata_path),
            },
            "labels_tsv": {
                "path": str(labels_path),
                "sha256": sha256_file(labels_path),
            },
        },
        "parameters": {
            "limit": args.limit,
            "selection_seed": args.selection_seed,
            "neighbor_seed": args.neighbor_seed,
            "baseline_top_k": args.baseline_top_k,
            "seeds_per_source": args.seeds_per_source,
            "max_seeds": args.max_seeds,
            "max_proposals_per_seed": args.max_proposals_per_seed,
            "max_neighbors_per_seed": args.max_neighbors_per_seed,
        },
        "hard_eligible_query_count": eligible_count,
        "selected_spec_names": selected,
        "query_count": len(queries),
        "candidate_count": len(candidates),
        "target_use": {
            "selection": "target connectivity absent from unchanged current+MolForge top-10",
            "generation": [],
            "forward_scoring": [],
            "ranking": [],
        },
    }
    write_prepare_artifacts(output_dir, queries, candidates, edits, targets, manifest)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "queries": len(queries),
                "candidates": len(candidates),
            }
        )
    )
    return 0


def command_score(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    queries = pd.read_csv(run_dir / "search_queries.csv")
    candidates = pd.read_csv(run_dir / "candidates.csv")
    forbidden = TARGET_COLUMNS.intersection(queries.columns).union(
        TARGET_COLUMNS.intersection(candidates.columns)
    )
    if forbidden:
        raise RuntimeError(
            f"Target fields are forbidden in score inputs: {sorted(forbidden)}"
        )
    ms_pred_root = Path(args.ms_pred_root).expanduser().resolve()
    python_path = Path(args.python_path).expanduser().resolve()
    gen_checkpoint = Path(args.gen_checkpoint).expanduser().resolve()
    inten_checkpoint = Path(args.inten_checkpoint).expanduser().resolve()
    ms_pred_info = verify_official_ms_pred(ms_pred_root, args.expected_ms_pred_commit)
    checkpoint_info = {
        "generator": {
            "path": str(gen_checkpoint),
            "sha256": sha256_file(gen_checkpoint),
        },
        "intensity": {
            "path": str(inten_checkpoint),
            "sha256": sha256_file(inten_checkpoint),
        },
    }
    if (
        args.expected_gen_sha256
        and checkpoint_info["generator"]["sha256"] != args.expected_gen_sha256
    ):
        raise RuntimeError("Generator checkpoint SHA-256 mismatch")
    if (
        args.expected_inten_sha256
        and checkpoint_info["intensity"]["sha256"] != args.expected_inten_sha256
    ):
        raise RuntimeError("Intensity checkpoint SHA-256 mismatch")

    output_dir = run_dir / "iceberg_scoring"
    output_dir.mkdir(exist_ok=False)
    all_scores: list[dict[str, Any]] = []
    query_stats = []
    for query in queries.to_dict(orient="records"):
        spec_name = query["spec_name"]
        query_candidates = candidates[candidates["query_spec_name"] == spec_name]
        prediction_path, wall_seconds = _run_official_iceberg(
            query=query,
            candidates=query_candidates,
            query_dir=output_dir / spec_name,
            ms_pred_root=ms_pred_root,
            python_path=python_path,
            gen_checkpoint=gen_checkpoint,
            inten_checkpoint=inten_checkpoint,
            collision_energies=tuple(args.collision_energies),
            gpu=args.gpu,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        score_rows, missing = _load_and_score_predictions(
            prediction_path=prediction_path,
            candidates=query_candidates,
            observed_path=Path(query["observed_spectrum_path"]),
            ms_pred_root=ms_pred_root,
        )
        all_scores.extend(score_rows)
        query_stats.append(
            {
                "spec_name": spec_name,
                "candidate_count": len(query_candidates),
                "missing_predictions": missing,
                "wall_seconds": wall_seconds,
                "prediction_path": str(prediction_path),
                "prediction_sha256": sha256_file(prediction_path),
            }
        )

    scores_path = output_dir / "iceberg_scores.csv"
    pd.DataFrame(all_scores).to_csv(scores_path, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "stage": "score",
        "ms_pred": ms_pred_info,
        "checkpoints": checkpoint_info,
        "parameters": {
            "collision_energies_ev": args.collision_energies,
            "collision_energy_type": "absolute_eV",
            "ppm": 20,
            "ignore_precursor": True,
            "energy_aggregation": "maximum_sparse_cosine",
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "gpu": args.gpu,
        },
        "iceberg_calls": len(query_stats),
        "iceberg_wall_seconds": sum(row["wall_seconds"] for row in query_stats),
        "query_stats": query_stats,
        "scores": {"path": str(scores_path), "sha256": sha256_file(scores_path)},
        "target_fields_used": [],
    }
    (output_dir / "score_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"scores": str(scores_path), "iceberg_calls": len(query_stats)}))
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir).expanduser().resolve()
    score_dir = run_dir / "iceberg_scoring"
    scores = pd.read_csv(score_dir / "iceberg_scores.csv")
    targets = pd.read_csv(run_dir / "evaluation_targets.csv")
    edits = pd.read_csv(run_dir / "edit_statistics.csv")
    score_manifest = json.loads((score_dir / "score_manifest.json").read_text())
    details, aggregate = evaluate_scores(scores, targets, edits, score_manifest)
    details_path = run_dir / "evaluation_details.csv"
    aggregate_path = run_dir / "evaluation.json"
    details.to_csv(details_path, index=False)
    aggregate["outputs"] = {
        "evaluation_details_csv": str(details_path),
        "evaluation_details_sha256": sha256_file(details_path),
    }
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--current-union", required=True)
    prepare.add_argument("--molforge-union", required=True)
    prepare.add_argument("--metadata-csv", required=True)
    prepare.add_argument("--labels-tsv", required=True)
    prepare.add_argument("--spec-dir", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--limit", type=int, choices=(4, 16, 64), required=True)
    prepare.add_argument("--selection-seed", type=int, default=13420260711)
    prepare.add_argument("--neighbor-seed", type=int, default=13420260711)
    prepare.add_argument("--baseline-top-k", type=int, default=10)
    prepare.add_argument("--seeds-per-source", type=int, default=4)
    prepare.add_argument("--max-seeds", type=int, default=8)
    prepare.add_argument("--max-proposals-per-seed", type=int, default=256)
    prepare.add_argument("--max-neighbors-per-seed", type=int, default=64)
    prepare.set_defaults(func=command_prepare)

    score = subparsers.add_parser("score")
    score.add_argument("--run-dir", required=True)
    score.add_argument("--ms-pred-root", required=True)
    score.add_argument("--python-path", required=True)
    score.add_argument("--gen-checkpoint", required=True)
    score.add_argument("--inten-checkpoint", required=True)
    score.add_argument("--expected-ms-pred-commit", default=DEFAULT_MS_PRED_COMMIT)
    score.add_argument("--expected-gen-sha256")
    score.add_argument("--expected-inten-sha256")
    score.add_argument(
        "--collision-energies",
        type=int,
        nargs="+",
        default=list(DEFAULT_COLLISION_ENERGIES),
    )
    score.add_argument("--gpu", type=int, required=True)
    score.add_argument("--batch-size", type=int, default=8)
    score.add_argument("--num-workers", type=int, default=6)
    score.set_defaults(func=command_score)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.set_defaults(func=command_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
