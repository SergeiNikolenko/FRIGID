#!/usr/bin/env python
"""Score a frozen RankLoop candidate pool with official ICEBERG predictions."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dlm.utils.benchmark_selection import load_spec_manifest  # noqa: E402
from frigid.rankloop_corpus import _git_revision, sha256_file  # noqa: E402
from frigid.rankloop_inference import (  # noqa: E402
    candidate_identity_sha256,
    load_inference_candidate_frame,
)
from run_gems_iceberg_diagnostic import (  # noqa: E402
    _load_and_score_predictions,
    _run_official_iceberg,
    connectivity_key,
    load_observed_query_labels,
    normalize_instrument,
    verify_official_ms_pred,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Target-blind ICEBERG forward scoring of frozen candidates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--spec-dir", required=True)
    parser.add_argument("--spec-manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ms-pred-root", required=True)
    parser.add_argument("--python-path", required=True)
    parser.add_argument("--gen-checkpoint", required=True)
    parser.add_argument("--inten-checkpoint", required=True)
    parser.add_argument("--expected-ms-pred-commit", required=True)
    parser.add_argument("--expected-gen-sha256", required=True)
    parser.add_argument("--expected-inten-sha256", required=True)
    parser.add_argument("--collision-energies", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def _python_executable_path(value: str) -> Path:
    """Return an absolute executable path without resolving a virtualenv symlink."""

    return Path(value).expanduser().absolute()


def _unsupported_iceberg_elements(
    smiles: str, valid_elements: set[str]
) -> tuple[str, ...]:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Cannot parse candidate SMILES: {smiles!r}")
    return tuple(
        sorted({atom.GetSymbol() for atom in mol.GetAtoms()} - valid_elements)
    )


def main() -> int:
    args = parse_args()
    if args.top_k <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("top_k and batch_size must be positive; num_workers cannot be negative.")
    candidate_path = Path(args.candidates).expanduser().resolve()
    spec_dir = Path(args.spec_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ms_pred_root = Path(args.ms_pred_root).expanduser().resolve()
    python_path = _python_executable_path(args.python_path)
    gen_checkpoint = Path(args.gen_checkpoint).expanduser().resolve()
    inten_checkpoint = Path(args.inten_checkpoint).expanduser().resolve()
    manifest_path = (
        Path(args.spec_manifest).expanduser().resolve() if args.spec_manifest else None
    )
    for name, path in {
        "candidate table": candidate_path,
        "ICEBERG Python": python_path,
        "generator checkpoint": gen_checkpoint,
        "intensity checkpoint": inten_checkpoint,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    if output_dir.exists():
        raise FileExistsError(f"Forward scoring output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    candidates = load_inference_candidate_frame(candidate_path)
    candidates["rank"] = pd.to_numeric(candidates["rank"], errors="raise").astype(int)
    available_queries = set(candidates["query_spec_name"])
    if manifest_path:
        query_names = load_spec_manifest(manifest_path)
    else:
        query_names = sorted(available_queries)
    missing_queries = [name for name in query_names if name not in available_queries]
    if missing_queries:
        raise ValueError(f"Candidate table is missing query {missing_queries[0]!r}.")

    order = {name: index for index, name in enumerate(query_names)}
    selected = candidates[candidates["query_spec_name"].isin(query_names)].copy()
    selected["query_order"] = selected["query_spec_name"].map(order).astype(int)
    selected = selected.sort_values(["query_order", "rank"], kind="mergesort")
    selected = selected.groupby("query_spec_name", sort=False).head(args.top_k).copy()
    selected["candidate_inchi_key_connectivity"] = selected["candidate_smiles"].map(
        connectivity_key
    )
    if selected["candidate_inchi_key_connectivity"].isna().any():
        bad = selected.loc[
            selected["candidate_inchi_key_connectivity"].isna(), "candidate_smiles"
        ].iloc[0]
        raise ValueError(f"Cannot derive candidate connectivity key: {bad!r}")
    selected["candidate_index"] = np.arange(len(selected), dtype=np.int64)
    reference = selected.drop(
        columns=["candidate_inchi_key_connectivity", "candidate_index", "query_order"]
    )
    input_identity = candidate_identity_sha256(reference)

    observed = load_observed_query_labels(spec_dir, query_names)
    observed_map = observed.set_index("spec", verify_integrity=True)
    ms_pred_info = verify_official_ms_pred(ms_pred_root, args.expected_ms_pred_commit)
    actual_gen_hash = sha256_file(gen_checkpoint)
    actual_inten_hash = sha256_file(inten_checkpoint)
    if actual_gen_hash != args.expected_gen_sha256:
        raise RuntimeError(f"ICEBERG generator hash mismatch: {actual_gen_hash}")
    if actual_inten_hash != args.expected_inten_sha256:
        raise RuntimeError(f"ICEBERG intensity hash mismatch: {actual_inten_hash}")

    sys.path.insert(0, str(ms_pred_root / "src"))
    from ms_pred.common.chem_utils import VALID_ELEMENTS

    valid_elements = set(VALID_ELEMENTS)

    score_rows: list[dict[str, object]] = []
    query_stats: list[dict[str, object]] = []
    started = time.perf_counter()
    predictions_root = output_dir / "predictions"
    predictions_root.mkdir()
    for query_name in query_names:
        rows = selected[selected["query_spec_name"] == query_name].copy()
        unsupported_by_candidate = rows["candidate_smiles"].map(
            lambda smiles: _unsupported_iceberg_elements(smiles, valid_elements)
        )
        supported_rows = rows.loc[unsupported_by_candidate.map(len).eq(0)].copy()
        unsupported_elements = sorted(
            {
                element
                for candidate_elements in unsupported_by_candidate
                for element in candidate_elements
            }
        )
        query = observed_map.loc[query_name]
        query_payload = {
            "spec_name": query_name,
            "ionization": str(query["ionization"]),
            "instrument": normalize_instrument(query["instrument"]),
        }
        query_dir = predictions_root / query_name
        observed_path = spec_dir / f"{query_name}.ms"
        if supported_rows.empty:
            prediction_path = None
            wall_seconds = 0.0
            missing_predictions = len(rows)
            rows_scored = [
                {
                    **candidate,
                    "iceberg_score": math.nan,
                    "best_collision_energy_ev": math.nan,
                    "precursor_mz": math.nan,
                }
                for candidate in rows.to_dict(orient="records")
            ]
        else:
            prediction_path, wall_seconds = _run_official_iceberg(
                query=query_payload,
                candidates=supported_rows,
                query_dir=query_dir,
                ms_pred_root=ms_pred_root,
                python_path=python_path,
                gen_checkpoint=gen_checkpoint,
                inten_checkpoint=inten_checkpoint,
                collision_energies=tuple(args.collision_energies),
                gpu=args.gpu,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            rows_scored, missing_predictions = _load_and_score_predictions(
                prediction_path=prediction_path,
                candidates=rows,
                observed_path=observed_path,
                ms_pred_root=ms_pred_root,
            )
        score_rows.extend(rows_scored)
        query_stats.append(
            {
                "spec_name": query_name,
                "candidate_count": len(rows),
                "iceberg_input_count": len(supported_rows),
                "unsupported_candidate_count": len(rows) - len(supported_rows),
                "unsupported_elements": unsupported_elements,
                "missing_predictions": missing_predictions,
                "wall_seconds": wall_seconds,
                "prediction_path": str(prediction_path) if prediction_path else None,
                "prediction_sha256": sha256_file(prediction_path) if prediction_path else None,
            }
        )

    scored = pd.DataFrame(score_rows).rename(columns={"iceberg_score": "forward_score"})
    if len(scored) != len(reference):
        raise AssertionError("Forward scoring did not preserve all selected candidates.")
    if candidate_identity_sha256(scored) != input_identity:
        raise AssertionError("Forward scoring changed the frozen candidate pool.")
    finite = np.isfinite(scored["forward_score"].to_numpy(dtype=np.float64))
    scored["forward_scored"] = finite
    reference_path = output_dir / "reference_candidates.csv"
    scores_path = output_dir / "forward_scores.csv"
    reference.to_csv(reference_path, index=False)
    scored.to_csv(scores_path, index=False)

    revision, dirty = _git_revision(PROJECT_ROOT)
    run_manifest = {
        "schema_version": 1,
        "state": "completed",
        "repo": {"commit": revision, "dirty": dirty},
        "parameters": vars(args),
        "target_use": "none",
        "inputs": {
            "candidates": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)},
            "spec_manifest": (
                {"path": str(manifest_path), "sha256": sha256_file(manifest_path)}
                if manifest_path
                else None
            ),
            "ms_pred": ms_pred_info,
            "generator_checkpoint": {"path": str(gen_checkpoint), "sha256": actual_gen_hash},
            "intensity_checkpoint": {"path": str(inten_checkpoint), "sha256": actual_inten_hash},
        },
        "candidate_pool": {
            "query_count": len(query_names),
            "candidate_count": len(reference),
            "identity_sha256": input_identity,
            "unchanged": True,
        },
        "forward_scores": {
            "finite_count": int(finite.sum()),
            "missing_count": int((~finite).sum()),
            "nondegenerate_query_count": int(
                scored.groupby("query_spec_name")["forward_score"].apply(
                    lambda values: values.notna().all() and values.max() - values.min() > 1e-8
                ).sum()
            ),
            "wall_seconds": time.perf_counter() - started,
            "query_stats": query_stats,
        },
        "outputs": {
            "reference_candidates": {"path": str(reference_path), "sha256": sha256_file(reference_path)},
            "forward_scores": {"path": str(scores_path), "sha256": sha256_file(scores_path)},
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(run_manifest["forward_scores"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
