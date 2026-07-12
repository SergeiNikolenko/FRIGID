import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from frigid.stoned_expansion import (
    connectivity_key,
    generate_stoned_candidates,
)


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_stoned_candidate_expansion.py"
SPEC = importlib.util.spec_from_file_location("run_stoned_candidate_expansion", SCRIPT_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)

SEED_SMILES = "CCOC(=O)NCC1CCCCC1"
SEED_FORMULA = "C10H19NO2"


def _generate(seed=13420260711, raw_proposals=512, max_accepted=64):
    return generate_stoned_candidates(
        SEED_SMILES,
        query_formula=SEED_FORMULA,
        seed=seed,
        raw_proposals=raw_proposals,
        mutation_depths=(1, 2),
        operations=("replacement",),
        max_accepted=max_accepted,
    )


def test_generation_is_deterministic_for_same_seed():
    first = _generate(raw_proposals=128)
    second = _generate(raw_proposals=128)
    assert first == second


def test_generation_changes_for_different_seed():
    first, first_proposals, _ = _generate(seed=1, raw_proposals=64)
    second, second_proposals, _ = _generate(seed=2, raw_proposals=64)
    assert first != second or first_proposals != second_proposals


def test_accepted_candidates_are_valid_connected_formula_exact_and_unique():
    candidates, proposals, stats = _generate()
    assert candidates
    assert len(proposals) == 512
    assert stats.proposals_considered == 512
    assert stats.accepted_unique == len(candidates)
    assert len(candidates) <= 64
    keys = [candidate.inchi_key_connectivity for candidate in candidates]
    assert len(keys) == len(set(keys))
    for candidate in candidates:
        mol = Chem.MolFromSmiles(candidate.smiles)
        assert mol is not None
        assert len(Chem.GetMolFrags(mol)) == 1
        assert rdMolDescriptors.CalcMolFormula(mol) == SEED_FORMULA
        assert connectivity_key(mol) == candidate.inchi_key_connectivity


def test_seed_and_original_pool_connectivity_are_excluded():
    initial, _, _ = _generate()
    assert initial
    excluded = {initial[0].inchi_key_connectivity}
    filtered, _, stats = generate_stoned_candidates(
        SEED_SMILES,
        query_formula=SEED_FORMULA,
        seed=13420260711,
        raw_proposals=512,
        max_accepted=64,
        exclude_connectivity_keys=excluded,
    )
    seed_key = connectivity_key(SEED_SMILES)
    assert seed_key not in {candidate.inchi_key_connectivity for candidate in filtered}
    assert excluded.isdisjoint(candidate.inchi_key_connectivity for candidate in filtered)
    assert stats.rejection_counts["original_or_duplicate_connectivity"] >= 1


def test_candidate_budget_is_strictly_enforced():
    candidates, proposals, stats = _generate(max_accepted=2)
    assert len(candidates) <= 2
    assert len(proposals) == 512
    assert stats.accepted_unique == len(candidates)


def test_target_fields_are_rejected_from_generation_queries(tmp_path):
    queries = tmp_path / "queries.csv"
    pd.DataFrame(
        [{"spec_name": "q1", "formula": "C2H6O", "target_smiles": "CCO"}]
    ).to_csv(queries, index=False)
    with pytest.raises(ValueError, match="Target fields are forbidden"):
        RUNNER.load_queries(queries, ["q1"])


def test_fixed_manifest_hash_mismatch_is_rejected(tmp_path):
    manifest = tmp_path / "fixed.tsv"
    manifest.write_text("spec_name\nq1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        RUNNER.load_fixed_manifest(manifest, "0" * 64)


def test_synthetic_known_neighbor_is_recovered():
    candidates, _, _ = generate_stoned_candidates(
        "CC(C)O",
        query_formula="C3H8O",
        seed=13420260711,
        raw_proposals=512,
        mutation_depths=(1, 2),
        operations=("replacement",),
        max_accepted=64,
    )
    recovered = {candidate.inchi_key_connectivity for candidate in candidates}
    assert connectivity_key("CCCO") in recovered


def test_invalid_settings_and_formula_mismatch_are_rejected():
    with pytest.raises(ValueError, match="raw_proposals"):
        generate_stoned_candidates(
            "CCO", query_formula="C2H6O", seed=1, raw_proposals=0
        )
    with pytest.raises(ValueError, match="Seed formula mismatch"):
        generate_stoned_candidates(
            "CCO", query_formula="C3H8O", seed=1, raw_proposals=1
        )


def test_abcd_comparison_recovers_synthetic_target_without_generation_target_fields():
    spec_name = "q1"
    union = pd.DataFrame(
        [(spec_name, 1, "CC(C)O", "union")],
        columns=("query_spec_name", "rank", "candidate_smiles", "source_name"),
    )
    baseline = RUNNER.build_baseline_pool(union, [spec_name])
    seed_record = RUNNER._candidate_record("CC(C)O")
    seeds = pd.DataFrame(
        [
            {
                "query_spec_name": spec_name,
                "seed_index": 1,
                "seed_smiles": "CC(C)O",
                "seed_inchi_key_connectivity": seed_record[
                    "candidate_inchi_key_connectivity"
                ],
                "seed_source": "union",
                "seed_source_rank": 1,
                "query_formula": "C3H8O",
            }
        ]
    )
    raw, accepted, _, _ = RUNNER.generate_expansions(
        seeds,
        {spec_name: {seed_record["candidate_inchi_key_connectivity"]}},
        global_seed=13420260711,
        raw_proposals_per_seed=512,
        accepted_per_seed=64,
        mutation_depths=(1, 2),
        operations=("replacement",),
    )
    assert RUNNER.TARGET_COLUMNS.isdisjoint(raw.columns)
    assert RUNNER.TARGET_COLUMNS.isdisjoint(accepted.columns)
    assert raw.groupby("comparison_role").size().to_dict() == {
        "stoned_C": 512,
        "stoned_D": 256,
    }
    accepted_counts = accepted.groupby("generator").size().to_dict()
    assert accepted_counts.get("stoned_C", 0) <= 64
    assert accepted_counts.get("two_switch_B", 0) <= 64
    assert accepted_counts.get("stoned_D", 0) <= 32
    assert accepted_counts.get("two_switch_D", 0) <= 32
    pools = RUNNER.build_variant_pools(baseline, accepted)
    target_record = RUNNER._candidate_record("CCCO")
    ranked = RUNNER.rank_variant_pools(
        pools, {spec_name: target_record["fingerprint"]}
    )
    targets = pd.DataFrame(
        [
            {
                "spec_name": spec_name,
                "target_smiles": target_record["candidate_smiles"],
                "target_inchi_key_connectivity": target_record[
                    "candidate_inchi_key_connectivity"
                ],
                "target_fingerprint": target_record["fingerprint"],
            }
        ]
    )
    metrics = RUNNER.evaluate_variants(ranked, targets).set_index("variant")
    assert metrics.loc["A_union", "candidate_recall"] == 0
    assert metrics.loc["C_union_stoned", "candidate_recall"] == 1


def test_seed_selection_backfills_unique_candidates_after_cross_source_duplicates():
    union = pd.DataFrame(
        [
            ("q1", 1, "CC(C)O", "union"),
            ("q1", 2, "CCCO", "union"),
            ("q1", 3, "CCOC", "union"),
        ],
        columns=("query_spec_name", "rank", "candidate_smiles", "source_name"),
    )
    molforge = pd.DataFrame(
        [
            ("q1", 1, "CC(C)O", "molforge"),
            ("q1", 2, "CCCO", "molforge"),
            ("q1", 3, "CCOC", "molforge"),
        ],
        columns=("query_spec_name", "rank", "candidate_smiles", "source_name"),
    )
    seeds, _, rejected = RUNNER.select_seeds(
        ["q1"],
        {"q1": "C3H8O"},
        [("union", union), ("molforge", molforge)],
        seeds_per_source=2,
        max_seeds=3,
    )
    assert len(seeds) == 3
    assert seeds["seed_inchi_key_connectivity"].nunique() == 3
    assert rejected["duplicate_seed"] >= 1
