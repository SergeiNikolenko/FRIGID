import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "autoresearch_marlin_clearml_score.py"
)
SPEC = importlib.util.spec_from_file_location(
    "autoresearch_marlin_clearml_score",
    SCRIPT,
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_extract_metrics_preserves_exact_metrics_and_builds_score():
    metrics = MODULE.extract_metrics(
        {
            "Molecular metrics": {
                "Exact@1": {"last": 0.2},
                "Exact@10": {"last": 0.3},
                "Candidate return": {"last": 0.9},
                "Validity": {"last": 0.8},
                "Mass validity": {"last": 0.7},
                "Uniqueness": {"last": 0.6},
            }
        }
    )

    assert metrics["exact_top1"] == pytest.approx(0.2)
    assert metrics["exact_top10"] == pytest.approx(0.3)
    assert metrics["exact_score"] == pytest.approx(0.24)
    assert metrics["tanimoto_top1"] == 0.0


def test_validate_tags_rejects_oracle_and_locked_test():
    with pytest.raises(ValueError, match="locked-test"):
        MODULE.validate_tags(
            MODULE.COMMON_REQUIRED_TAGS | {"selection-validation", "locked-test"}
        )
    with pytest.raises(ValueError, match="oracle"):
        MODULE.validate_tags(
            MODULE.COMMON_REQUIRED_TAGS
            | {"selection-validation", "oracle-target-formula"}
        )


def test_validate_tags_requires_selection_contract():
    with pytest.raises(ValueError, match="selection-validation"):
        MODULE.validate_tags(MODULE.COMMON_REQUIRED_TAGS)


def test_validate_tags_accepts_only_locked_test_in_final_mode():
    final_tags = MODULE.COMMON_REQUIRED_TAGS | {"heldout-nplib1"}
    MODULE.validate_tags(final_tags, "final")

    with pytest.raises(ValueError, match="selection-validation"):
        MODULE.validate_tags(final_tags | {"selection-validation"}, "final")
