from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "calibrate_rankloop_conformal.py"
SPEC = importlib.util.spec_from_file_location("calibrate_rankloop_conformal", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["calibrate_rankloop_conformal"] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_panel(root: Path, prefix: str, count: int, absent_last: bool = False) -> dict:
    names = [f"{prefix}{index}" for index in range(count)]
    manifest_rows = []
    target_rows = []
    candidate_rows = []
    for index, name in enumerate(names):
        target = f"TARGET{prefix}{index}"
        manifest_rows.append(
            {"spec_name": name, "inchikey_first_block": target, "formula": "C2H6"}
        )
        target_rows.append({"spec_name": name, "target_inchi_key": target})
        candidate_ids = [target, f"ALT{prefix}{index}A", f"ALT{prefix}{index}B"]
        if absent_last and index == count - 1:
            candidate_ids[0] = f"ALT{prefix}{index}C"
        if index % 2:
            candidate_ids[0], candidate_ids[1] = candidate_ids[1], candidate_ids[0]
        for rank, (candidate_id, score) in enumerate(
            zip(candidate_ids, (0.8, 0.5, 0.2)), start=1
        ):
            candidate_rows.append(
                {
                    "query_spec_name": name,
                    "rank": rank,
                    "candidate_smiles": f"C{index}{rank}",
                    "candidate_inchi_key_first_block": candidate_id,
                    "tanimoto_to_mist": score,
                }
            )
    manifest = root / f"{prefix}_manifest.tsv"
    targets = root / f"{prefix}_targets.csv"
    candidates = root / f"{prefix}_candidates.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest, sep="\t", index=False)
    pd.DataFrame(target_rows).to_csv(targets, index=False)
    pd.DataFrame(candidate_rows).to_csv(candidates, index=False)
    return {"manifest": manifest, "targets": targets, "candidates": candidates}


def _write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "alpha": 0.2,
                "coverage_scope": "conditional_on_target_in_frozen_candidate_pool",
                "score_column": "tanimoto_to_mist",
                "temperature_grid": [0.1, 0.5, 1.0],
                "temperature_fit_fraction": 0.5,
                "split_seed": 7,
                "minimum_temperature_targets": 2,
                "minimum_conformal_targets": 2,
                "raps_lambda": 0.01,
                "raps_k_reg": 2,
                "confidence_group_quantile": 0.5,
                "minimum_group_targets": 2,
                "maximum_calibration_risk": 0.6,
                "minimum_abstention_queries": 2,
            }
        ),
        encoding="utf-8",
    )


def test_finite_sample_quantile_uses_conservative_rank() -> None:
    result = MODULE.finite_sample_quantile([0.1, 0.2, 0.3, 0.4], alpha=0.2)

    assert result["finite_sample_rank"] == 4
    assert result["value"] == 0.4


def test_flat_scores_keep_frozen_rank_for_prediction_set() -> None:
    query = MODULE.QueryCandidates(
        spec_name="q",
        target_id="B",
        candidate_ids=("A", "B", "C"),
        candidate_smiles=("A", "B", "C"),
        scores=np.array([1.0, 1.0, 1.0]),
    )

    scores = MODULE.nonconformity_scores(query, temperature=1.0)

    assert np.allclose(scores, [1 / 3, 2 / 3, 1.0])
    assert MODULE.prediction_set_size(scores, threshold=0.5) == 2
    assert query.candidate_ids[:2] == ("A", "B")


def test_rejects_calibration_evaluation_query_overlap() -> None:
    query = MODULE.QueryCandidates(
        spec_name="same",
        target_id="A",
        candidate_ids=("A", "B"),
        candidate_smiles=("A", "B"),
        scores=np.array([0.7, 0.3]),
    )

    with pytest.raises(ValueError, match="query overlap"):
        MODULE.distribution_audit([query], [query], temperature=1.0)


def test_end_to_end_keeps_targets_out_of_candidate_set_output(tmp_path: Path) -> None:
    calibration = _write_panel(tmp_path, "cal", 12)
    evaluation = _write_panel(tmp_path, "eval", 4, absent_last=True)
    config = tmp_path / "config.json"
    _write_config(config)
    output = tmp_path / "output"
    args = SimpleNamespace(
        config=config,
        calibration_candidates=calibration["candidates"],
        calibration_targets=calibration["targets"],
        calibration_manifest=calibration["manifest"],
        evaluation_candidates=evaluation["candidates"],
        evaluation_targets=evaluation["targets"],
        evaluation_manifest=evaluation["manifest"],
        output_dir=output,
    )

    result = MODULE.run(args)

    candidate_sets = pd.read_csv(output / "candidate_sets.csv")
    metrics = pd.read_csv(output / "query_metrics.csv")
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert len(candidate_sets) == 16
    assert "target_present" not in candidate_sets.columns
    assert "covered" not in candidate_sets.columns
    assert set(metrics.groupby("policy")["target_present"].sum()) == {3}
    assert result["summary"]["ranking_unchanged"] is True
    assert manifest["underlying_ranking_changed"] is False
    assert manifest["target_fields_used"]["inference"] == []
    assert json.loads((output / "calibration.json").read_text())


def test_rejects_manifest_target_identity_mismatch(tmp_path: Path) -> None:
    panel = _write_panel(tmp_path, "bad", 2)
    manifest = pd.read_csv(panel["manifest"], sep="\t")
    manifest.loc[0, "inchikey_first_block"] = "WRONG"
    manifest.to_csv(panel["manifest"], sep="\t", index=False)

    with pytest.raises(ValueError, match="Target identity disagrees"):
        MODULE.load_queries(
            panel["candidates"],
            panel["targets"],
            panel["manifest"],
            "tanimoto_to_mist",
        )
