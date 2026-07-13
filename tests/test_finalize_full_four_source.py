import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).parents[1] / "scripts"
SCRIPT_PATH = SCRIPT_DIR / "finalize_full_four_source.py"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location("finalize_full_four_source", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


def recorded(path: Path, **extra) -> dict:
    return {"path": str(path), "sha256": MODULE.sha256_file(path), **extra}


def test_load_shard_list_requires_exact_unique_directories(tmp_path):
    shards = []
    for index in range(2):
        shard = tmp_path / f"shard_{index}"
        shard.mkdir()
        shards.append(shard)
    shard_file = tmp_path / "shards.txt"
    shard_file.write_text("\n".join(map(str, shards)) + "\n")

    assert MODULE.load_shard_list(shard_file, expected_count=2) == shards

    shard_file.write_text(f"{shards[0]}\n{shards[0]}\n")
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.load_shard_list(shard_file, expected_count=2)


def test_validate_recorded_file_rejects_hash_drift(tmp_path):
    artifact = tmp_path / "artifact.csv"
    artifact.write_text("a\n")
    record = recorded(artifact)
    artifact.write_text("changed\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        MODULE.validate_recorded_file(record, "artifact")


def test_resolve_retrieval_inputs_validates_nested_manifests(tmp_path):
    expected_sha = "a" * 64
    candidates = tmp_path / "candidate_scores.csv"
    candidates.write_text("query_spec_name,rank,candidate_smiles\nq1,1,C\n")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("spec_name,target_smiles,target_inchi_key\nq1,C,KEY\n")
    fingerprints = tmp_path / "fingerprints.npz"
    fingerprints.write_bytes(b"fingerprints")
    export_manifest = tmp_path / "export_manifest.json"
    write_json(
        export_manifest,
        {
            "purpose": "mist_fingerprint_export",
            "status": "completed",
            "code": {"dirty": False},
            "inputs": {"spec_manifest": {"sha256": expected_sha}},
            "outputs": {
                "metadata_csv": recorded(metadata, row_count=17082),
                "fingerprints_npz": recorded(
                    fingerprints, shape=[17082, 4096]
                ),
            },
        },
    )
    manifest_path = tmp_path / "retrieval_manifest.json"
    write_json(
        manifest_path,
        {
            "purpose": "frigid_full_train_only_retrieval",
            "status": "completed",
            "exit_code": 0,
            "code": {"dirty": False},
            "expected_manifest": {"sha256": expected_sha},
            "outputs": {
                "candidate_scores": recorded(candidates, row_count=170820),
                "mist_export_manifest": recorded(export_manifest),
            },
        },
    )

    assert MODULE.resolve_retrieval_inputs(manifest_path, expected_sha) == (
        candidates,
        metadata,
        fingerprints,
    )


def test_resolve_molforge_candidates_requires_full_conversion(tmp_path):
    expected_sha = "b" * 64
    candidates = tmp_path / "molforge.csv"
    candidates.write_text("query_spec_name,rank,candidate_smiles\nq1,1,C\n")
    converter = tmp_path / "converter.json"
    write_json(
        converter,
        {
            "query_count": 17082,
            "spec_manifest": {"sha256": expected_sha},
            "target_fields_used": [],
        },
    )
    manifest_path = tmp_path / "molforge_manifest.json"
    write_json(
        manifest_path,
        {
            "purpose": "frigid_full_molforge_suffix",
            "status": "completed",
            "exit_code": 0,
            "code": {"dirty": False},
            "inputs": {"expected_manifest": {"sha256": expected_sha}},
            "selection": {"start_index": 8630, "max_spectra": 8452},
            "outputs": {
                "full_candidates": recorded(candidates),
                "converter_manifest": recorded(converter),
            },
        },
    )

    assert MODULE.resolve_molforge_candidates(manifest_path, expected_sha) == candidates
