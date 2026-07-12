import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from frigid.forward_consistency import fuse_forward_consistency_scores
from frigid.rankloop_inference import candidate_identity_sha256


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "score_rankloop_forward_consistency.py"
SPEC = importlib.util.spec_from_file_location("score_rankloop_forward_consistency", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _frame(forward_scores=(0.1, 0.9, 0.2)):
    return pd.DataFrame(
        {
            "query_spec_name": ["query-1"] * 3,
            "candidate_smiles": ["CC", "CCC", "CCCC"],
            "rank": [1, 2, 3],
            "tanimoto_to_mist": [0.9, 0.8, 0.7],
            "forward_score": list(forward_scores),
        }
    )


def test_python_executable_path_preserves_virtualenv_symlink(tmp_path):
    base_python = tmp_path / "base-python"
    base_python.touch()
    venv_python = tmp_path / "venv-python"
    venv_python.symlink_to(base_python)

    executable = MODULE._python_executable_path(str(venv_python))

    assert executable == venv_python
    assert executable.resolve() == base_python


def test_unsupported_iceberg_elements_are_detected_without_target_metadata():
    valid_elements = {"C", "H", "N", "O"}

    assert MODULE._unsupported_iceberg_elements("CCO", valid_elements) == ()
    assert MODULE._unsupported_iceberg_elements("[Ge]", valid_elements) == ("Ge",)


def test_forward_blend_can_promote_consistent_candidate_without_identity_change():
    frame = _frame()
    expected_identity = candidate_identity_sha256(frame)

    ranked = fuse_forward_consistency_scores(
        frame,
        alpha=0.8,
        normalization="zscore",
        mode="blend",
    )

    assert ranked.iloc[0]["candidate_smiles"] == "CCC"
    assert ranked["rankloop_rank"].tolist() == [1, 2, 3]
    assert candidate_identity_sha256(ranked) == expected_identity


def test_forward_contradiction_penalizes_only_low_forward_tail():
    ranked = fuse_forward_consistency_scores(
        _frame(),
        alpha=1.0,
        normalization="rank",
        mode="contradiction",
        contradiction_quantile=0.5,
    )

    assert ranked.iloc[0]["candidate_smiles"] == "CCC"
    assert ranked.iloc[-1]["candidate_smiles"] == "CCCC"


def test_missing_forward_score_falls_back_to_reference_order():
    ranked = fuse_forward_consistency_scores(
        _frame((0.1, np.nan, 0.2)),
        alpha=1.0,
        normalization="zscore",
        mode="blend",
    )

    assert ranked["candidate_smiles"].tolist() == ["CC", "CCC", "CCCC"]
    assert ranked["forward_fallback"].all()


def test_degenerate_forward_scores_fall_back_to_reference_order():
    ranked = fuse_forward_consistency_scores(
        _frame((0.5, 0.5, 0.5)),
        alpha=1.0,
        normalization="rank",
        mode="blend",
    )

    assert ranked["candidate_smiles"].tolist() == ["CC", "CCC", "CCCC"]
    assert ranked["forward_fallback"].all()
