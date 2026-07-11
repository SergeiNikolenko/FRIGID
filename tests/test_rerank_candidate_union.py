import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_DIR = Path(__file__).parents[1] / "scripts"
FUSION_PATH = SCRIPT_DIR / "fuse_candidate_sources.py"
RERANK_PATH = SCRIPT_DIR / "rerank_candidate_union.py"
sys.path.insert(0, str(SCRIPT_DIR))

FUSION_SPEC = importlib.util.spec_from_file_location(
    "fuse_candidate_sources", FUSION_PATH
)
FUSION = importlib.util.module_from_spec(FUSION_SPEC)
assert FUSION_SPEC.loader is not None
sys.modules["fuse_candidate_sources"] = FUSION
FUSION_SPEC.loader.exec_module(FUSION)

RERANK_SPEC = importlib.util.spec_from_file_location(
    "rerank_candidate_union", RERANK_PATH
)
RERANK = importlib.util.module_from_spec(RERANK_SPEC)
assert RERANK_SPEC.loader is not None
sys.modules["rerank_candidate_union"] = RERANK
RERANK_SPEC.loader.exec_module(RERANK)


def candidate(
    *,
    query: str,
    source: str,
    source_rank: int,
    smiles: str,
    source_priority: int,
):
    fingerprint = np.zeros(128, dtype=np.float32)
    fingerprint[{"CC": 1, "CCC": 2, "CCCC": 3}[smiles]] = 1.0
    return FUSION.SourceCandidate(
        query_spec_name=query,
        source_name=source,
        source_rank=source_rank,
        smiles=smiles,
        candidate_spec_name=None,
        inchi_key_first_block={
            "CC": "OTMSDBZUPAUEDD",
            "CCC": "ATUOYWHBWRKTHZ",
            "CCCC": "IJDNQMDRQITEOD",
        }[smiles],
        fingerprint=fingerprint,
        source_priority=source_priority,
    )


def ranked_smiles(rows):
    return [row.ranked.candidate_smiles for row in rows]


def test_predeclared_score_family_is_small_and_exact():
    assert [config.formula_bonus for config in RERANK.RERANK_CONFIGS] == [
        0.0,
        0.01,
        0.025,
        0.05,
        0.0,
        0.0,
        0.0,
        0.01,
        0.025,
        0.05,
    ]
    assert [config.consensus_bonus for config in RERANK.RERANK_CONFIGS] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.02,
        0.05,
        0.1,
        0.02,
        0.05,
        0.1,
    ]


def test_mist_only_is_equivalent_to_default_fusion_with_same_tie_break(
    monkeypatch,
):
    monkeypatch.setattr(
        RERANK,
        "compute_candidate_formula",
        lambda smiles: {"CC": "C2H6", "CCC": "C3H8"}[smiles],
    )
    query_fp = np.ones(128, dtype=np.float32)
    raw = [
        candidate(
            query="q1",
            source="control",
            source_rank=1,
            smiles="CC",
            source_priority=0,
        ),
        candidate(
            query="q1",
            source="temperature",
            source_rank=1,
            smiles="CCC",
            source_priority=1,
        ),
        candidate(
            query="q1",
            source="retrieval",
            source_rank=2,
            smiles="CC",
            source_priority=2,
        ),
    ]
    legacy = FUSION.rank_query_candidates(
        query_fp,
        FUSION.deduplicate_by_inchikey_first_block(raw),
        top_k=10,
    )
    features = RERANK.build_candidate_features(
        query_fp=query_fp,
        raw_candidates=raw,
        query_formula="C2H6",
        source_names=["control", "temperature", "retrieval"],
    )
    sidecar = RERANK.rerank_candidate_features(
        features,
        config=RERANK.BASELINE_CONFIG,
        top_k=10,
    )

    assert ranked_smiles(sidecar) == [row.candidate_smiles for row in legacy]
    assert [row.ranked.source_name for row in sidecar] == [
        row.source_name for row in legacy
    ]
    assert [row.ranked.tanimoto_to_mist for row in sidecar] == [
        row.tanimoto_to_mist for row in legacy
    ]


def test_formula_comes_from_manifest_and_candidate_smiles_not_source_columns(
    monkeypatch, tmp_path
):
    manifest_path = tmp_path / "manifest.tsv"
    pd.DataFrame([{"spec_name": "q1", "formula": "C2H6"}]).to_csv(
        manifest_path, sep="\t", index=False
    )
    source_path = tmp_path / "source.csv"
    pd.DataFrame(
        [
            {
                "query_spec_name": "q1",
                "query_formula": "CH4",
                "rank": 1,
                "candidate_smiles": "CC",
                "candidate_formula": "WRONG",
            }
        ]
    ).to_csv(source_path, index=False)

    observed_formula_smiles = []

    def fake_formula(smiles):
        observed_formula_smiles.append(smiles)
        return "C2H6"

    monkeypatch.setattr(
        FUSION,
        "compute_morgan_fingerprint",
        lambda smiles, bits, radius: np.ones(bits, dtype=np.float32),
    )
    monkeypatch.setattr(
        FUSION,
        "_compute_inchi_key_first_block",
        lambda smiles: "OTMSDBZUPAUEDD",
    )
    monkeypatch.setattr(RERANK, "compute_candidate_formula", fake_formula)

    _, formulas = RERANK.load_manifest_formulas(manifest_path)
    rows = FUSION._load_source_rows(
        source_name="source",
        source_path=source_path,
        source_priority=0,
        fingerprint_bits=128,
        fingerprint_radius=2,
    )
    features = RERANK.build_candidate_features(
        query_fp=np.ones(128, dtype=np.float32),
        raw_candidates=rows,
        query_formula=formulas["q1"],
        source_names=["source"],
    )

    assert features[0].candidate_formula == "C2H6"
    assert features[0].formula_match is True
    assert observed_formula_smiles == ["CC"]


def test_cross_source_support_and_neighborhood_are_computed_before_deduplication(
    monkeypatch,
):
    monkeypatch.setattr(
        RERANK,
        "compute_candidate_formula",
        lambda smiles: {"CC": "C2H6", "CCC": "C3H8"}[smiles],
    )
    query_fp = np.ones(128, dtype=np.float32)
    raw = [
        candidate(
            query="q1",
            source="control",
            source_rank=1,
            smiles="CC",
            source_priority=0,
        ),
        candidate(
            query="q1",
            source="temperature",
            source_rank=4,
            smiles="CC",
            source_priority=1,
        ),
        candidate(
            query="q1",
            source="retrieval",
            source_rank=1,
            smiles="CCC",
            source_priority=2,
        ),
    ]
    features = RERANK.build_candidate_features(
        query_fp=query_fp,
        raw_candidates=raw,
        query_formula="C2H6",
        source_names=["control", "temperature", "retrieval"],
    )
    ethane = next(row for row in features if row.candidate_formula == "C2H6")

    assert len(features) == 2
    assert ethane.source_support_count == 2
    assert ethane.source_support_fraction == 0.5
    assert ethane.cross_source_neighborhood >= ethane.source_support_fraction
    assert ethane.consensus_signal > 0.0


def test_target_changes_cannot_change_target_blind_rankings(monkeypatch):
    monkeypatch.setattr(
        RERANK,
        "compute_candidate_formula",
        lambda smiles: {"CC": "C2H6", "CCC": "C3H8"}[smiles],
    )

    def fake_fingerprint(smiles, bits, radius):
        fingerprint = np.zeros(bits, dtype=np.float32)
        fingerprint[{"CC": 1, "CCCC": 3}[smiles]] = 1.0
        return fingerprint

    monkeypatch.setattr(RERANK, "compute_morgan_fingerprint", fake_fingerprint)
    query_fp = np.ones((1, 128), dtype=np.float32)
    raw = [
        candidate(
            query="q1",
            source="control",
            source_rank=1,
            smiles="CC",
            source_priority=0,
        ),
        candidate(
            query="q1",
            source="retrieval",
            source_rank=1,
            smiles="CCC",
            source_priority=1,
        ),
    ]
    rankings = RERANK.build_target_blind_rankings(
        query_order=["q1"],
        mist_binary=query_fp,
        manifest_formulas={"q1": "C2H6"},
        source_rows_by_query={"q1": raw},
        source_names=["control", "retrieval"],
        configs=(RERANK.RERANK_CONFIGS[1],),
        top_k=2,
    )
    before = ranked_smiles(rankings["formula_0p01"]["q1"])

    target_a = {"q1": {"target_smiles": "CC", "target_inchi_key": "OTMSDBZUPAUEDD"}}
    target_b = {"q1": {"target_smiles": "CCCC", "target_inchi_key": "IJDNQMDRQITEOD"}}
    metrics_a = RERANK.evaluate_queries(
        ["q1"], rankings["formula_0p01"], target_a, 2, 128, 2
    )
    metrics_b = RERANK.evaluate_queries(
        ["q1"], rankings["formula_0p01"], target_b, 2, 128, 2
    )
    after = ranked_smiles(rankings["formula_0p01"]["q1"])

    assert before == after
    assert metrics_a != metrics_b


def test_dev_selection_uses_declared_lexicographic_objective_and_first_tie():
    configs = (
        RERANK.BASELINE_CONFIG,
        RERANK.RerankConfig("first", 0.01, 0.0),
        RERANK.RerankConfig("second", 0.025, 0.0),
        RERANK.RerankConfig("third", 0.05, 0.0),
    )
    metrics = {
        "mist_only": {
            "tanimoto_top10": 0.4,
            "tanimoto_top1": 0.3,
            "exact_match_top10": 0.2,
        },
        "first": {
            "tanimoto_top10": 0.5,
            "tanimoto_top1": 0.3,
            "exact_match_top10": 0.2,
        },
        "second": {
            "tanimoto_top10": 0.5,
            "tanimoto_top1": 0.31,
            "exact_match_top10": 0.1,
        },
        "third": {
            "tanimoto_top10": 0.5,
            "tanimoto_top1": 0.31,
            "exact_match_top10": 0.1,
        },
    }

    selected = RERANK.select_dev_configuration(metrics, configs=configs)

    assert selected.name == "second"


def test_manifest_and_metadata_order_must_match(tmp_path):
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {"spec_name": "q2", "smiles": "CC", "inchi_key": "A"},
            {"spec_name": "q1", "smiles": "CCC", "inchi_key": "B"},
        ]
    ).to_csv(metadata, index=False)
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(
        [
            {"spec_name": "q1", "formula": "C2H6"},
            {"spec_name": "q2", "formula": "C3H8"},
        ]
    ).to_csv(manifest, sep="\t", index=False)
    fingerprints = tmp_path / "fingerprints.npz"
    np.savez_compressed(fingerprints, mist_binary=np.zeros((2, 128), dtype=np.float32))

    with pytest.raises(ValueError, match="exactly match"):
        RERANK.run_reranker_evaluation(
            source_specs=[],
            mist_metadata_csv=metadata,
            mist_fingerprints_npz=fingerprints,
            benchmark_manifest=manifest,
            output_dir=tmp_path / "out",
            selection_count=1,
            top_k=2,
            fingerprint_bits=128,
            fingerprint_radius=2,
        )
