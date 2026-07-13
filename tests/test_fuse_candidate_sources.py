import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "fuse_candidate_sources.py"
SPEC = importlib.util.spec_from_file_location("fuse_candidate_sources", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(SCRIPT_PATH.parent))
sys.modules["fuse_candidate_sources"] = MODULE
SPEC.loader.exec_module(MODULE)


def test_infer_source_schema_supports_canonical_and_dlm_layouts():
    canonical = pd.DataFrame({"spec_name": ["s1"], "rank": [1], "smiles": ["C"]})
    query_col, rank_col, smiles_col = MODULE.infer_source_schema(canonical)
    assert query_col == "spec_name"
    assert rank_col == "rank"
    assert smiles_col == "smiles"

    dlm_like = pd.DataFrame(
        {
            "query_spec_name": ["s1"],
            "rank": [1],
            "candidate_smiles": ["C"],
        }
    )
    query_col, rank_col, smiles_col = MODULE.infer_source_schema(dlm_like)
    assert query_col == "query_spec_name"
    assert rank_col == "rank"
    assert smiles_col == "candidate_smiles"


def test_deduplicate_by_inchikey_across_sources_keeps_deterministic_best():
    first = MODULE.SourceCandidate(
        query_spec_name="q1",
        source_name="beta",
        source_rank=2,
        smiles="C1",
        candidate_spec_name=None,
        inchi_key_first_block="KEY",
        fingerprint=np.array([0.0, 1.0], dtype=np.float32),
    )
    second = MODULE.SourceCandidate(
        query_spec_name="q1",
        source_name="alpha",
        source_rank=1,
        smiles="C2",
        candidate_spec_name=None,
        inchi_key_first_block="KEY",
        fingerprint=np.array([1.0, 0.0], dtype=np.float32),
    )
    deduped = MODULE.deduplicate_by_inchikey_first_block([first, second])
    assert len(deduped) == 1
    assert deduped[0].source_name == "alpha"
    assert deduped[0].smiles == "C2"


def test_rank_tie_break_is_deterministic_by_source_name():
    query_fp = np.array([1.0, 1.0], dtype=np.float32)
    candidates = [
        MODULE.SourceCandidate(
            query_spec_name="q1",
            source_name="model_b",
            source_rank=1,
            smiles="CC",
            candidate_spec_name=None,
            inchi_key_first_block="A",
            fingerprint=np.array([1.0, 0.0], dtype=np.float32),
        ),
        MODULE.SourceCandidate(
            query_spec_name="q1",
            source_name="model_a",
            source_rank=1,
            smiles="CCC",
            candidate_spec_name=None,
            inchi_key_first_block="B",
            fingerprint=np.array([0.0, 1.0], dtype=np.float32),
        ),
    ]
    ranked = MODULE.rank_query_candidates(query_fp, candidates, top_k=2)
    assert [row.candidate_smiles for row in ranked] == ["CCC", "CC"]


def test_rank_tie_prefers_later_source_priority():
    query_fp = np.array([1.0, 1.0], dtype=np.float32)
    candidates = [
        MODULE.SourceCandidate(
            query_spec_name="q1",
            source_name="control",
            source_rank=1,
            smiles="CC",
            candidate_spec_name=None,
            inchi_key_first_block="A",
            fingerprint=np.array([1.0, 0.0], dtype=np.float32),
            source_priority=0,
        ),
        MODULE.SourceCandidate(
            query_spec_name="q1",
            source_name="retrieval",
            source_rank=1,
            smiles="CCC",
            candidate_spec_name=None,
            inchi_key_first_block="B",
            fingerprint=np.array([0.0, 1.0], dtype=np.float32),
            source_priority=1,
        ),
    ]

    ranked = MODULE.rank_query_candidates(query_fp, candidates, top_k=2)

    assert [row.source_name for row in ranked] == ["retrieval", "control"]


def test_no_target_smiles_leaks_into_ranking(monkeypatch, tmp_path):
    metadata = pd.DataFrame(
        [
            {
                "spec_name": "q1",
                "target_smiles": "target_1",
                "target_inchi_key": "T1",
            },
            {
                "spec_name": "q2",
                "target_smiles": "target_2",
                "target_inchi_key": "T2",
            },
        ]
    )
    metadata_path = tmp_path / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    np.savez_compressed(
        tmp_path / "fingerprints.npz",
        mist_binary=np.array(
            [[1.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        ),
    )
    npz_path = tmp_path / "fingerprints.npz"

    source = pd.DataFrame(
        [
            {
                "query_spec_name": "q1",
                "rank": 1,
                "candidate_smiles": "cand_a",
            },
            {
                "query_spec_name": "q1",
                "rank": 1,
                "candidate_smiles": "cand_b",
            },
            {
                "query_spec_name": "q2",
                "rank": 1,
                "candidate_smiles": "cand_a",
            },
            {
                "query_spec_name": "q2",
                "rank": 1,
                "candidate_smiles": "cand_b",
            },
        ]
    )
    source_path = tmp_path / "source.csv"
    source.to_csv(source_path, index=False)

    def fake_compute_morgan_fingerprint(smiles: str, bits: int, radius: int):
        if smiles == "cand_a":
            return np.array([1.0, 0.0], dtype=np.float32)
        if smiles == "cand_b":
            return np.array([0.0, 1.0], dtype=np.float32)
        if smiles == "target_1":
            return np.array([1.0, 0.0], dtype=np.float32)
        if smiles == "target_2":
            return np.array([0.0, 1.0], dtype=np.float32)
        raise AssertionError(f"unexpected smiles {smiles!r}")

    monkeypatch.setattr(MODULE, "compute_morgan_fingerprint", fake_compute_morgan_fingerprint)
    monkeypatch.setattr(
        MODULE,
        "_compute_inchi_key_first_block",
        lambda smiles: {"cand_a": "A", "cand_b": "B"}[smiles],
    )

    predictions, _, detailed = MODULE.run_fuse_candidate_sources(
        source_specs=[("src", source_path)],
        mist_metadata_csv=metadata_path,
        mist_fingerprints_npz=npz_path,
        output_dir=tmp_path / "out",
        top_k=2,
        fingerprint_bits=2,
        fingerprint_radius=2,
        source_contributions=True,
    )

    first_prediction = predictions.set_index("name").loc["q1", "pred_smiles_1"]
    second_prediction = predictions.set_index("name").loc["q2", "pred_smiles_1"]
    assert first_prediction == second_prediction == "cand_a"
    assert detailed["fingerprint_source"].unique().tolist() == ["mist_binary"]
    assert detailed["candidate_method"].unique().tolist() == ["fused_candidates"]
    contributions = pd.read_csv(tmp_path / "out/source_contributions.csv").fillna("")
    assert set(contributions["variant"]) == {
        "full",
        "source_only",
        "without_source",
    }
    assert len(contributions) == 6
    run_manifest = json.loads((tmp_path / "out/run_manifest.json").read_text())
    assert run_manifest["ranking_contract"]["target_fields_used_by_ranking"] == []


def test_top10_tanimoto_is_max_not_mean():
    ranked = [
        MODULE.RankedCandidate(
            rank=1,
            source_name="s",
            source_rank=1,
            source_candidate_rank=1,
            candidate_smiles="CC",
            candidate_spec_name=None,
            candidate_inchi_key_first_block="Q1",
            tanimoto_to_mist=0.2,
            fingerprint=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ),
        MODULE.RankedCandidate(
            rank=2,
            source_name="s",
            source_rank=1,
            source_candidate_rank=2,
            candidate_smiles="CCC",
            candidate_spec_name=None,
            candidate_inchi_key_first_block="Q2",
            tanimoto_to_mist=0.8,
            fingerprint=np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        ),
    ]

    metrics = MODULE.evaluate_ranked_predictions(
        target_inchi_key_first_block="Q0",
        target_fingerprint=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        ranked=ranked,
        top_k=2,
    )

    assert metrics["tanimoto_top1"] == 0.25
    assert metrics["tanimoto_top10"] == 0.75


def test_rejects_misaligned_metadata_and_mist_rows(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        [{"spec_name": "q1", "target_smiles": "C", "target_inchi_key": "KEY"}]
    ).to_csv(metadata_path, index=False)
    npz_path = tmp_path / "fingerprints.npz"
    np.savez_compressed(npz_path, mist_binary=np.zeros((2, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="different row counts"):
        MODULE.run_fuse_candidate_sources(
            source_specs=[],
            mist_metadata_csv=metadata_path,
            mist_fingerprints_npz=npz_path,
            output_dir=tmp_path / "out",
            top_k=2,
            fingerprint_bits=4,
            fingerprint_radius=2,
        )


def test_rejects_source_query_absent_from_mist_metadata(monkeypatch, tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        [{"spec_name": "q1", "target_smiles": "C", "target_inchi_key": "KEY"}]
    ).to_csv(metadata_path, index=False)
    npz_path = tmp_path / "fingerprints.npz"
    np.savez_compressed(npz_path, mist_binary=np.zeros((1, 4), dtype=np.float32))
    source_path = tmp_path / "source.csv"
    source_path.write_text("query_spec_name,rank,candidate_smiles\nunknown,1,C\n")

    monkeypatch.setattr(
        MODULE,
        "_load_source_rows",
        lambda **_: [
            MODULE.SourceCandidate(
                query_spec_name="unknown",
                source_name="source",
                source_rank=1,
                smiles="C",
                candidate_spec_name=None,
                inchi_key_first_block="KEY",
                fingerprint=np.zeros(4, dtype=np.float32),
            )
        ],
    )

    with pytest.raises(ValueError, match="absent from MIST metadata"):
        MODULE.run_fuse_candidate_sources(
            source_specs=[("source", source_path)],
            mist_metadata_csv=metadata_path,
            mist_fingerprints_npz=npz_path,
            output_dir=tmp_path / "out",
            top_k=10,
            fingerprint_bits=4,
            fingerprint_radius=2,
        )
