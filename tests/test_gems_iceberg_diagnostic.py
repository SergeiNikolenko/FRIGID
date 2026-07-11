import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_gems_iceberg_diagnostic.py"
SPEC = importlib.util.spec_from_file_location(
    "run_gems_iceberg_diagnostic", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _candidate_frame(rows):
    return pd.DataFrame(
        rows,
        columns=("query_spec_name", "rank", "candidate_smiles"),
    )


def test_hard_query_selection_is_seeded_after_target_absence_filter():
    metadata = pd.DataFrame(
        {
            "spec_name": ["q1", "q2", "q3"],
            "smiles": ["CC", "CCC", "CCCC"],
            "inchi_key_first_block": [
                MODULE.connectivity_key("CC"),
                MODULE.connectivity_key("CCC"),
                MODULE.connectivity_key("CCCC"),
            ],
        }
    )
    current = _candidate_frame(
        [
            ("q1", 1, "CC"),
            ("q2", 1, "CCO"),
            ("q3", 1, "CCN"),
        ]
    )
    molforge = _candidate_frame(
        [
            ("q1", 1, "CO"),
            ("q2", 1, "COC"),
            ("q3", 1, "CNC"),
        ]
    )

    selected, targets, eligible_count = MODULE.select_hard_queries(
        metadata,
        current,
        molforge,
        limit=2,
        selection_seed=134,
    )
    repeated, _, _ = MODULE.select_hard_queries(
        metadata,
        current,
        molforge,
        limit=2,
        selection_seed=134,
    )

    assert "q1" not in selected
    assert selected == repeated
    assert eligible_count == 2
    assert targets["spec_name"].tolist() == selected


def test_sparse_cosine_uses_20ppm_assignment_and_ignores_precursor():
    observed = np.array([[100.0, 1.0], [200.0, 0.5], [499.5, 10.0]])
    matching = np.array([[100.001, 1.0], [199.999, 0.5], [499.5, 0.0]])
    nonmatching = np.array([[110.0, 1.0], [210.0, 0.5], [499.5, 100.0]])

    matching_score = MODULE.sparse_cosine_similarity_20ppm(
        matching, observed, precursor_mz=500.0
    )
    nonmatching_score = MODULE.sparse_cosine_similarity_20ppm(
        nonmatching, observed, precursor_mz=500.0
    )

    assert matching_score > 0.99
    assert nonmatching_score == 0.0


def test_search_space_has_no_target_fields_and_is_deterministic(tmp_path):
    spec_dir = tmp_path / "spec"
    spec_dir.mkdir()
    (spec_dir / "q.ms").write_text(">compound q\n\n>ms2peaks\n100 1\n")
    current = _candidate_frame(
        [
            ("q", 1, "CCOC(=O)NCC1CCCCC1"),
            ("q", 2, "CCNC(=O)OCC1CCCCC1"),
        ]
    )
    molforge = _candidate_frame(
        [
            ("q", 1, "COC(=O)NCCC1CCCCC1"),
            ("q", 2, "CCOC(=O)NCC1CCCCC1"),
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "spec": "q",
                "formula": "C10H19NO2",
                "ionization": "[M+H]+",
                "instrument": "Orbitrap",
            }
        ]
    )
    kwargs = {
        "spec_dir": spec_dir,
        "baseline_top_k": 10,
        "seeds_per_source": 2,
        "max_seeds": 4,
        "neighbor_seed": 134,
        "max_proposals_per_seed": 24,
        "max_neighbors_per_seed": 4,
    }

    first = MODULE.build_search_space(["q"], current, molforge, labels, **kwargs)
    second = MODULE.build_search_space(["q"], current, molforge, labels, **kwargs)

    for frame in first[:2]:
        assert MODULE.TARGET_COLUMNS.isdisjoint(frame.columns)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    assert first[1]["is_neighbor"].sum() > 0


def test_score_evaluation_ranking_is_target_blind_until_metrics():
    target_key = MODULE.connectivity_key("CCO")
    other_key = MODULE.connectivity_key("CCN")
    scores = pd.DataFrame(
        [
            {
                "query_spec_name": "q",
                "candidate_index": 0,
                "candidate_smiles": "CCN",
                "candidate_inchi_key_connectivity": other_key,
                "is_neighbor": 0,
                "iceberg_score": 0.8,
            },
            {
                "query_spec_name": "q",
                "candidate_index": 1,
                "candidate_smiles": "CCO",
                "candidate_inchi_key_connectivity": target_key,
                "is_neighbor": 1,
                "iceberg_score": 0.9,
            },
        ]
    )
    targets = pd.DataFrame(
        [
            {
                "spec_name": "q",
                "target_smiles": "CCO",
                "target_inchi_key_connectivity": target_key,
            }
        ]
    )
    edits = pd.DataFrame(
        [
            {
                "accepted_unique": 1,
                "proposals_considered": 2,
                "invalid_counts_json": '{"sanitize_failure": 1}',
            }
        ]
    )
    manifest = {"iceberg_calls": 1, "iceberg_wall_seconds": 2.0}

    details, aggregate = MODULE.evaluate_scores(scores, targets, edits, manifest)

    assert details.loc[0, "candidate_recovery_before"] == 0
    assert details.loc[0, "candidate_recovery_after"] == 1
    assert details.loc[0, "exact_match_top10"] == 1
    assert aggregate["new_candidate_recoveries"] == 1
    assert aggregate["target_fields_used_by_generation_or_scoring"] == []
