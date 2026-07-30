from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).parents[1] / "scripts/autoresearch_marlin_score.py"
SPEC = importlib.util.spec_from_file_location("autoresearch_marlin_score", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_aggregate_seed_metrics_uses_fixed_weighted_score() -> None:
    first = {name: 1.0 for name in MODULE.SCORE_WEIGHTS}
    second = {name: 0.0 for name in MODULE.SCORE_WEIGHTS}
    result = MODULE.aggregate_seed_metrics([first, second])

    assert result["marlin_validation_score"] == 0.5
    assert result["research_stage"] == "exact_retrieval"
    assert result["research_score"] == 3.5
    assert result["seed_count"] == 2


def test_aggregate_seed_metrics_treats_nan_as_zero() -> None:
    metrics = {name: 0.0 for name in MODULE.SCORE_WEIGHTS}
    metrics["tanimoto_top1"] = math.nan

    result = MODULE.aggregate_seed_metrics([metrics])

    assert result["tanimoto_top1"] == 0.0
    assert result["marlin_validation_score"] == 0.0


def test_aggregate_seed_metrics_uses_prerequisite_gate_before_exact() -> None:
    metrics = {name: 0.0 for name in MODULE.SCORE_WEIGHTS}
    metrics["validity"] = 0.25

    result = MODULE.aggregate_seed_metrics([metrics])

    assert result["research_stage"] == "valid_decoding"
    assert result["research_score"] == 0.25


def test_protected_evaluator_includes_metric_and_fixed_panels() -> None:
    protected = set(MODULE.PROTECTED_EVALUATOR_PATHS)

    assert "src/marlin/research_metric.py" in protected
    assert "src/marlin/benchmark_selection.py" in protected
    assert "configs/benchmarks/nplib1_v1/" in protected


def test_require_clean_evaluator_rejects_protected_changes() -> None:
    completed = type("Completed", (), {"stdout": " M src/marlin/research_metric.py"})()
    with patch.object(MODULE.subprocess, "run", return_value=completed):
        try:
            MODULE.require_clean_evaluator()
        except RuntimeError as error:
            assert "research_metric.py" in str(error)
        else:
            raise AssertionError("modified research metric was accepted")
