import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "prepare_selective_ttt_neighbors.py"
)
SPEC = importlib.util.spec_from_file_location(
    "prepare_selective_ttt_neighbors",
    SCRIPT_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["prepare_selective_ttt_neighbors"] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fingerprints(rows: list[tuple[int, ...]]) -> np.ndarray:
    values = np.zeros((len(rows), MODULE.FINGERPRINT_BITS), dtype=np.uint8)
    for row_index, active_bits in enumerate(rows):
        values[row_index, list(active_bits)] = 1
    return values


def _write_valid_inputs(tmp_path: Path) -> dict[str, Path | int]:
    train_library_csv = tmp_path / "train.csv"
    train_fingerprints_npz = tmp_path / "train.npz"
    query_index_csv = tmp_path / "queries.csv"
    query_fingerprints_npz = tmp_path / "queries.npz"
    exclusion_tokens_csv = tmp_path / "exclusions.csv"
    mist_checkpoint = tmp_path / "mist.ckpt"
    base_dlm_checkpoint = tmp_path / "base.ckpt"

    _write_csv(
        train_library_csv,
        MODULE.TRAIN_FIELDS,
        [
            {
                "spec_name": "train_overlap",
                "split": "train",
                "smiles": "CC",
                "inchi_key_first_block": "AAAAAAAAAAAAAA",
            },
            {
                "spec_name": "train_best",
                "split": "train",
                "smiles": "CCC",
                "inchi_key_first_block": "BBBBBBBBBBBBBB",
            },
            {
                "spec_name": "train_a_tie",
                "split": "train",
                "smiles": "CO",
                "inchi_key_first_block": "CCCCCCCCCCCCCC",
            },
            {
                "spec_name": "train_b_tie",
                "split": "train",
                "smiles": "CN",
                "inchi_key_first_block": "DDDDDDDDDDDDDD",
            },
        ],
    )
    np.savez(
        train_fingerprints_npz,
        mist_binary=_fingerprints([(0, 1), (0, 1, 2), (0, 3), (0, 4)]),
    )
    _write_csv(query_index_csv, MODULE.QUERY_FIELDS, [{"spec_name": "query_1"}])
    np.savez(query_fingerprints_npz, mist_binary=_fingerprints([(0, 1)]))
    _write_csv(
        exclusion_tokens_csv,
        MODULE.EXCLUSION_FIELDS,
        [
            {
                "spec_name": "query_1",
                "connectivity_exclusion_token": MODULE.connectivity_exclusion_token(
                    "AAAAAAAAAAAAAA"
                ),
            }
        ],
    )
    mist_checkpoint.write_bytes(b"mist-checkpoint")
    base_dlm_checkpoint.write_bytes(b"base-dlm-checkpoint")
    return {
        "train_library_csv": train_library_csv,
        "train_fingerprints_npz": train_fingerprints_npz,
        "query_index_csv": query_index_csv,
        "query_fingerprints_npz": query_fingerprints_npz,
        "query_exclusion_tokens_csv": exclusion_tokens_csv,
        "mist_checkpoint": mist_checkpoint,
        "base_dlm_checkpoint": base_dlm_checkpoint,
        "output_dir": tmp_path / "output",
        "top_k": 2,
    }


def test_prepares_ranked_train_only_neighbors_and_provenance(tmp_path):
    inputs = _write_valid_inputs(tmp_path)

    manifest = MODULE.prepare_selective_ttt_neighbors(**inputs)

    with (inputs["output_dir"] / "neighbors.csv").open(encoding="utf-8") as handle:
        neighbors = list(csv.DictReader(handle))
    assert [row["train_spec_name"] for row in neighbors] == [
        "train_best",
        "train_a_tie",
    ]
    assert "train_overlap" not in {row["train_spec_name"] for row in neighbors}
    assert float(neighbors[0]["tanimoto_to_query_mist"]) == pytest.approx(2 / 3)

    bundle = json.loads(
        (inputs["output_dir"] / "query_bundles.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    isolation = bundle["state_isolation_contract"]
    assert isolation["scope"] == "query_local"
    assert isolation["initialize_from_base_checkpoint_for_every_query"] is True
    assert isolation["cross_query_learned_state_reuse_allowed"] is False
    assert isolation["adapter_execution_implemented"] is False
    assert isolation["prior_query_state_inputs"] == []
    assert len(isolation["planned_query_state_id"]) == 64
    assert isolation["base_dlm_checkpoint_sha256"] == MODULE.sha256_file(
        inputs["base_dlm_checkpoint"]
    )
    assert bundle["target_fields_available"] is False

    assert manifest["safeguards"]["target_fields_used_for_ranking"] == []
    assert (
        manifest["safeguards"][
            "target_derived_exclusion_token_used_only_for_filtering"
        ]
        is True
    )
    assert manifest["safeguards"]["raw_query_inchi_key_allowed"] is False
    assert manifest["inputs"]["mist_checkpoint"]["sha256"] == MODULE.sha256_file(
        inputs["mist_checkpoint"]
    )


def test_fails_closed_on_raw_query_target_columns(tmp_path):
    inputs = _write_valid_inputs(tmp_path)
    _write_csv(
        inputs["query_index_csv"],
        ("spec_name", "target_smiles"),
        [{"spec_name": "query_1", "target_smiles": "CC"}],
    )

    with pytest.raises(ValueError, match="columns must be exactly"):
        MODULE.prepare_selective_ttt_neighbors(**inputs)


def test_fails_closed_on_query_target_array(tmp_path):
    inputs = _write_valid_inputs(tmp_path)
    query_fingerprints = _fingerprints([(0, 1)])
    np.savez(
        inputs["query_fingerprints_npz"],
        mist_binary=query_fingerprints,
        ground_truth=query_fingerprints,
    )

    with pytest.raises(ValueError, match="NPZ arrays must be exactly"):
        MODULE.prepare_selective_ttt_neighbors(**inputs)


def test_fails_closed_when_train_library_contains_test_rows(tmp_path):
    inputs = _write_valid_inputs(tmp_path)
    with inputs["train_library_csv"].open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["split"] = "test"
    _write_csv(inputs["train_library_csv"], MODULE.TRAIN_FIELDS, rows)

    with pytest.raises(ValueError, match="only split=train"):
        MODULE.prepare_selective_ttt_neighbors(**inputs)


def test_requires_exclusion_token_for_every_query(tmp_path):
    inputs = _write_valid_inputs(tmp_path)
    _write_csv(
        inputs["query_exclusion_tokens_csv"],
        MODULE.EXCLUSION_FIELDS,
        [
            {
                "spec_name": "different_query",
                "connectivity_exclusion_token": MODULE.connectivity_exclusion_token(
                    "AAAAAAAAAAAAAA"
                ),
            }
        ],
    )

    with pytest.raises(ValueError, match="must match the query index exactly"):
        MODULE.prepare_selective_ttt_neighbors(**inputs)
