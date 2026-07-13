import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts/evaluate_mces_predictions.py"
SPEC = importlib.util.spec_from_file_location("evaluate_mces_predictions", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["evaluate_mces_predictions"] = MODULE
SPEC.loader.exec_module(MODULE)


def prediction_frame(names: list[str], targets: list[str]) -> pd.DataFrame:
    rows = []
    for name, target in zip(names, targets):
        row = {"name": name, "true_smiles": target}
        row.update({f"pred_smiles_{k}": "C" * k for k in range(1, 11)})
        rows.append(row)
    return pd.DataFrame(rows)


def test_load_prediction_frames_requires_locked_order_and_shared_targets(tmp_path):
    expected = ["q2", "q1"]
    control = tmp_path / "control.csv"
    union = tmp_path / "union.csv"
    prediction_frame(expected, ["CC", "CCC"]).to_csv(control, index=False)
    prediction_frame(expected, ["CC", "CCC"]).to_csv(union, index=False)

    variants, frames = MODULE.load_prediction_frames(
        [("control", control), ("union", union)], expected
    )

    assert variants == ["control", "union"]
    assert frames["control"]["name"].tolist() == expected

    prediction_frame(list(reversed(expected)), ["CCC", "CC"]).to_csv(
        union, index=False
    )
    with pytest.raises(ValueError, match="locked manifest order"):
        MODULE.load_prediction_frames(
            [("control", control), ("union", union)], expected
        )


def test_run_metric_task_exports_all_mces_prefixes():
    def fake_metric(*args, **kwargs):
        assert kwargs == {
            "solver": "PULP_CBC_CMD",
            "doMCES": True,
            "doFull": False,
            "filter_formula": False,
        }
        metrics = {f"mces@{k}": float(11 - k) for k in range(1, 11)}
        return metrics, [], [], [], []

    row = MODULE.run_metric_task(
        fake_metric,
        "q1",
        "union",
        "CC",
        ["CCC"],
    )

    assert row["spec_name"] == "q1"
    assert row["variant"] == "union"
    assert row["mces@1"] == 10.0
    assert row["mces@10"] == 1.0


def test_parse_prediction_argument_rejects_missing_name_path_separator():
    with pytest.raises(Exception, match="NAME=PATH"):
        MODULE.parse_prediction_argument("predictions.csv")
