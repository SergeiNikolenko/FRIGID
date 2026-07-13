#!/usr/bin/env python
"""Merge, fuse, and compare the frozen full four-source FRIGID reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from compare_paired_benchmark_runs import (
    compare_runs,
    load_results,
    write_outputs as write_comparison_outputs,
)
from fuse_candidate_sources import run_fuse_candidate_sources
from merge_dlm_benchmark_shards import merge_shards


SOURCE_ORDER = ("control", "temperature", "retrieval", "molforge0p172")
QUALITY_METRICS = (
    "tanimoto_top1",
    "tanimoto_top10",
    "exact_match_top1",
    "exact_match_top10",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def load_shard_list(path: Path, expected_count: int = 18) -> list[Path]:
    shard_dirs = [
        Path(line.strip()).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(shard_dirs) != expected_count:
        raise ValueError(
            f"Expected {expected_count} shard directories in {path}, "
            f"found {len(shard_dirs)}"
        )
    if len(shard_dirs) != len(set(shard_dirs)):
        raise ValueError(f"Shard list contains duplicate directories: {path}")
    missing = [shard for shard in shard_dirs if not shard.is_dir()]
    if missing:
        raise ValueError(f"Shard directories do not exist: {missing[:5]}")
    return shard_dirs


def validate_recorded_file(record: dict[str, Any], purpose: str) -> Path:
    path_value = record.get("path")
    expected_sha256 = record.get("sha256")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"Missing recorded path for {purpose}")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"Missing recorded SHA-256 for {purpose}")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Recorded {purpose} file does not exist: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{purpose} SHA-256 mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    return path


def resolve_retrieval_inputs(
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> tuple[Path, Path, Path]:
    manifest = load_json(manifest_path)
    if manifest.get("purpose") != "frigid_full_train_only_retrieval":
        raise ValueError(f"Unexpected retrieval purpose: {manifest_path}")
    if manifest.get("status") != "completed" or manifest.get("exit_code") != 0:
        raise ValueError(f"Retrieval run is not completed: {manifest_path}")
    if manifest.get("code", {}).get("dirty") is not False:
        raise ValueError(f"Retrieval run used a dirty checkout: {manifest_path}")
    if manifest.get("expected_manifest", {}).get("sha256") != expected_manifest_sha256:
        raise ValueError("Retrieval locked-manifest SHA-256 mismatch")

    outputs = manifest.get("outputs", {})
    candidates = validate_recorded_file(
        outputs.get("candidate_scores", {}), "retrieval candidates"
    )
    if outputs["candidate_scores"].get("row_count") != 170820:
        raise ValueError("Retrieval candidates do not contain exactly 170,820 rows")
    export_manifest_path = validate_recorded_file(
        outputs.get("mist_export_manifest", {}), "MIST export manifest"
    )
    export_manifest = load_json(export_manifest_path)
    if export_manifest.get("purpose") != "mist_fingerprint_export":
        raise ValueError("Unexpected MIST export purpose")
    if export_manifest.get("status") != "completed":
        raise ValueError("MIST export is not completed")
    if export_manifest.get("code", {}).get("dirty") is not False:
        raise ValueError("MIST export used a dirty checkout")
    spec_manifest = export_manifest.get("inputs", {}).get("spec_manifest", {})
    if spec_manifest.get("sha256") != expected_manifest_sha256:
        raise ValueError("MIST export locked-manifest SHA-256 mismatch")

    export_outputs = export_manifest.get("outputs", {})
    metadata = validate_recorded_file(
        export_outputs.get("metadata_csv", {}), "MIST metadata"
    )
    fingerprints = validate_recorded_file(
        export_outputs.get("fingerprints_npz", {}), "MIST fingerprints"
    )
    if export_outputs["metadata_csv"].get("row_count") != 17082:
        raise ValueError("MIST metadata does not contain 17,082 rows")
    if export_outputs["fingerprints_npz"].get("shape") != [17082, 4096]:
        raise ValueError("MIST fingerprints do not have shape [17082, 4096]")
    return candidates, metadata, fingerprints


def resolve_molforge_candidates(
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> Path:
    manifest = load_json(manifest_path)
    if manifest.get("purpose") != "frigid_full_molforge_suffix":
        raise ValueError(f"Unexpected MolForge purpose: {manifest_path}")
    if manifest.get("status") != "completed" or manifest.get("exit_code") != 0:
        raise ValueError(f"MolForge suffix run is not completed: {manifest_path}")
    if manifest.get("code", {}).get("dirty") is not False:
        raise ValueError(f"MolForge suffix used a dirty checkout: {manifest_path}")
    if manifest.get("inputs", {}).get("expected_manifest", {}).get(
        "sha256"
    ) != expected_manifest_sha256:
        raise ValueError("MolForge locked-manifest SHA-256 mismatch")
    selection = manifest.get("selection", {})
    if selection.get("start_index") != 8630 or selection.get("max_spectra") != 8452:
        raise ValueError("MolForge suffix range mismatch")

    outputs = manifest.get("outputs", {})
    candidates = validate_recorded_file(
        outputs.get("full_candidates", {}), "MolForge full candidates"
    )
    converter_manifest_path = validate_recorded_file(
        outputs.get("converter_manifest", {}), "MolForge converter manifest"
    )
    converter_manifest = load_json(converter_manifest_path)
    if converter_manifest.get("query_count") != 17082:
        raise ValueError("MolForge converter did not validate 17,082 queries")
    if converter_manifest.get("spec_manifest", {}).get(
        "sha256"
    ) != expected_manifest_sha256:
        raise ValueError("MolForge converter locked-manifest SHA-256 mismatch")
    if converter_manifest.get("target_fields_used") != []:
        raise ValueError("MolForge converter target-field contract mismatch")
    return candidates


def git_state(project_root: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "-C", project_root, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            [
                "git",
                "-C",
                project_root,
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def finalize_full_four_source(
    *,
    control_shards: list[Path],
    temperature_shards: list[Path],
    control_shards_file: Path,
    temperature_shards_file: Path,
    expected_manifest: Path,
    retrieval_manifest: Path,
    molforge_manifest: Path,
    output_dir: Path,
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError(f"Finalization output already exists: {output_dir}")
    expected_manifest_sha256 = sha256_file(expected_manifest)
    expected = pd.read_csv(expected_manifest, sep="\t", dtype=str).fillna("")
    expected_names = expected["spec_name"].astype(str).tolist()
    if len(expected_names) != 17082 or len(expected_names) != len(set(expected_names)):
        raise ValueError("Expected manifest must contain 17,082 unique spectra")

    retrieval_candidates, mist_metadata, mist_fingerprints = resolve_retrieval_inputs(
        retrieval_manifest, expected_manifest_sha256
    )
    molforge_candidates = resolve_molforge_candidates(
        molforge_manifest, expected_manifest_sha256
    )
    project_root = Path(__file__).resolve().parents[1]
    commit, dirty = git_state(project_root)
    if dirty:
        raise ValueError(f"Finalization checkout must be clean: {project_root}")

    output_dir.mkdir(parents=True, exist_ok=False)
    state_path = output_dir / "FINALIZE_MANIFEST.json"
    state: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "frigid_full_four_source_finalization",
        "status": "running",
        "start_timestamp": datetime.now(timezone.utc).isoformat(),
        "code": {
            "path": str(project_root),
            "commit": commit,
            "dirty": False,
        },
        "expected_manifest": {
            "path": str(expected_manifest),
            "sha256": expected_manifest_sha256,
            "query_count": len(expected_names),
        },
        "source_order": list(SOURCE_ORDER),
        "target_fields_used_by_ranking": [],
        "bootstrap": {
            "unit": "molecule",
            "resamples": bootstrap_resamples,
            "seed": seed,
        },
        "inputs": {
            "control_shards_file": {
                "path": str(control_shards_file),
                "sha256": sha256_file(control_shards_file),
            },
            "temperature_shards_file": {
                "path": str(temperature_shards_file),
                "sha256": sha256_file(temperature_shards_file),
            },
            "retrieval_manifest": {
                "path": str(retrieval_manifest),
                "sha256": sha256_file(retrieval_manifest),
            },
            "molforge_manifest": {
                "path": str(molforge_manifest),
                "sha256": sha256_file(molforge_manifest),
            },
        },
        "outputs": {},
        "pending_metrics": ["MCES"],
    }
    write_state(state_path, state)

    try:
        control_dir = output_dir / "control_merged"
        temperature_dir = output_dir / "temperature_merged"
        control_merge = merge_shards(
            control_shards, expected_manifest, control_dir
        )
        temperature_merge = merge_shards(
            temperature_shards, expected_manifest, temperature_dir
        )
        if control_merge["frozen_signature"]["source_name"] != (
            "dlm_control_no_ngboost100"
        ):
            raise ValueError("Unexpected control source signature")
        if temperature_merge["frozen_signature"]["source_name"] != (
            "dlm_temperature_no_ngboost200_temp0p8"
        ):
            raise ValueError("Unexpected temperature source signature")

        source_specs = [
            (
                "control",
                control_dir / "prediction_scores_mist_binary.csv",
            ),
            (
                "temperature",
                temperature_dir / "prediction_scores_mist_binary.csv",
            ),
            ("retrieval", retrieval_candidates),
            ("molforge0p172", molforge_candidates),
        ]
        fusion_dir = output_dir / "four_source_union"
        run_fuse_candidate_sources(
            source_specs=source_specs,
            mist_metadata_csv=mist_metadata,
            mist_fingerprints_npz=mist_fingerprints,
            output_dir=fusion_dir,
            top_k=10,
            fingerprint_bits=4096,
            fingerprint_radius=2,
            source_contributions=True,
        )

        fusion_details = pd.read_csv(fusion_dir / "detailed_results.csv", dtype=str)
        if fusion_details["spec_name"].astype(str).tolist() != expected_names:
            raise ValueError("Four-source fusion does not match locked manifest order")
        aggregate = load_json(fusion_dir / "aggregate_statistics.json")
        if aggregate.get("n_queries") != 17082:
            raise ValueError("Four-source fusion query count mismatch")
        if aggregate.get("source_names") != list(SOURCE_ORDER):
            raise ValueError("Four-source fusion source order mismatch")
        if aggregate.get("target_fields_used_by_ranking") != []:
            raise ValueError("Four-source fusion target-field contract mismatch")

        reference = load_results(control_dir / "detailed_results.csv", None)
        candidate = load_results(fusion_dir / "detailed_results.csv", None)
        comparison, paired = compare_runs(
            reference=reference,
            candidate=candidate,
            metrics=QUALITY_METRICS,
            bootstrap_resamples=bootstrap_resamples,
            confidence=0.95,
            seed=seed,
            bootstrap_unit="molecule",
            cluster_column="target_inchi_key",
        )
        comparison.update(
            {
                "reference": {
                    "name": "dlm_control_no_ngboost100",
                    "path": str(control_dir / "detailed_results.csv"),
                },
                "candidate": {
                    "name": "frozen_four_source_union",
                    "path": str(fusion_dir / "detailed_results.csv"),
                },
            }
        )
        comparison_dir = output_dir / "paired_comparison"
        write_comparison_outputs(comparison_dir, comparison, paired)

        outputs = {}
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file() or path == state_path:
                continue
            outputs[str(path.relative_to(output_dir))] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        state["status"] = "completed"
        state["exit_code"] = 0
        state["end_timestamp"] = datetime.now(timezone.utc).isoformat()
        state["outputs"] = outputs
        state["quality_result"] = comparison
        state["candidate_recall_exact"] = aggregate["candidate_recall_exact"]
        write_state(state_path, state)
        return state
    except Exception as exc:
        state["status"] = "failed"
        state["exit_code"] = 1
        state["end_timestamp"] = datetime.now(timezone.utc).isoformat()
        state["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_state(state_path, state)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-shards-file", required=True, type=Path)
    parser.add_argument("--temperature-shards-file", required=True, type=Path)
    parser.add_argument("--expected-manifest", required=True, type=Path)
    parser.add_argument("--retrieval-manifest", required=True, type=Path)
    parser.add_argument("--molforge-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    control_shards_file = args.control_shards_file.expanduser().resolve()
    temperature_shards_file = args.temperature_shards_file.expanduser().resolve()
    finalize_full_four_source(
        control_shards=load_shard_list(control_shards_file),
        temperature_shards=load_shard_list(temperature_shards_file),
        control_shards_file=control_shards_file,
        temperature_shards_file=temperature_shards_file,
        expected_manifest=args.expected_manifest.expanduser().resolve(),
        retrieval_manifest=args.retrieval_manifest.expanduser().resolve(),
        molforge_manifest=args.molforge_manifest.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
