from pathlib import Path
import sys
from types import SimpleNamespace

from scripts.evaluate_marlin_nplib1 import (
    _clearml_candidate_table,
    add_formula_metrics,
    formula_metric_summary,
    publish_clearml_evaluation,
)


def test_safe_import_precedes_rdkit_draw_on_faro() -> None:
    source = Path("scripts/evaluate_marlin_nplib1.py").read_text()

    assert source.index(
        "from dlm.utils.utils_chem import safe_to_smiles"
    ) < source.index("from rdkit import Chem, DataStructs")
    assert "from rdkit.Chem import AllChem, Draw, rdMolDescriptors" not in source


def test_add_formula_metrics_checks_first_ten_candidates():
    row = {
        "target_smiles": "CCO",
        "candidates": [
            {"formula": "CH4"},
            {"formula": "C2H6O"},
        ],
    }

    add_formula_metrics(row)

    assert row["formula_top1"] is False
    assert row["formula_top10"] is True


def test_formula_metric_summary_uses_all_rows_and_returned_rows():
    rows = [
        {
            "candidate_returned": True,
            "formula_top1": False,
            "formula_top10": True,
        },
        {
            "candidate_returned": False,
            "formula_top1": False,
            "formula_top10": False,
        },
    ]

    assert formula_metric_summary(rows) == {
        "formula_top1_all": 0.0,
        "formula_top1_returned": 0.0,
        "formula_top10_all": 0.5,
        "formula_top10_returned": 1.0,
    }


def test_clearml_candidate_table_represents_spectrum_without_candidates():
    table = _clearml_candidate_table(
        [
            {
                "spec_name": "spectrum-empty",
                "target_smiles": "CCO",
                "candidates": [],
            }
        ]
    )

    assert table.to_dict("records") == [
        {
            "spec_name": "spectrum-empty",
            "rank": None,
            "target_smiles": "CCO",
            "candidate_smiles": None,
            "exact": False,
            "formula_match": False,
            "target_tanimoto": None,
            "mass_error_ppm": None,
        }
    ]


def test_publish_clearml_evaluation_is_disabled_without_explicit_configuration(
    monkeypatch,
):
    monkeypatch.setitem(sys.modules, "clearml", None)

    assert (
        publish_clearml_evaluation(
            project_name=None,
            task_name=None,
            tags=[],
            metrics={},
            rows=[],
            settings={},
        )
        is None
    )


def test_publish_clearml_evaluation_reports_molecular_scalars_and_table(
    monkeypatch,
):
    scalar_calls = []
    table_calls = []
    connected = []
    closed = []

    class FakeLogger:
        def report_scalar(self, **kwargs):
            scalar_calls.append(kwargs)

        def report_table(self, **kwargs):
            table_calls.append(kwargs)

    class FakeTask:
        TaskTypes = SimpleNamespace(testing="testing")

        @staticmethod
        def init(**kwargs):
            task = SimpleNamespace(
                id="task-id",
                name=kwargs["task_name"],
                get_logger=lambda: FakeLogger(),
                connect=lambda value, name: connected.append((name, value)),
                get_output_log_web_page=lambda: "https://clearml.example/task-id",
                close=lambda: closed.append(True),
            )
            return task

    monkeypatch.setitem(sys.modules, "clearml", SimpleNamespace(Task=FakeTask))
    metrics = {
        "rows": 2,
        "exact_top1": 0.5,
        "exact_top10": 1.0,
        "formula_top1_all": 0.0,
        "formula_top10_all": 0.5,
        "candidate_return_rate": 1.0,
        "validity": 0.75,
        "mass_validity": 0.5,
        "uniqueness": 0.25,
        "tanimoto_top1": 0.4,
        "tanimoto_top10": float("nan"),
    }
    rows = [
        {
            "spec_name": "spectrum-1",
            "target_smiles": "CCO",
            "candidates": [
                {
                    "smiles": "CCN",
                    "formula": "C2H7N",
                    "exact_connectivity": False,
                    "target_fingerprint_tanimoto": 0.6,
                    "mass_error_ppm": 2.5,
                },
                {
                    "smiles": "CCO",
                    "formula": "C2H6O",
                    "exact_connectivity": True,
                    "target_fingerprint_tanimoto": 1.0,
                    "mass_error_ppm": 0.1,
                },
            ],
        }
    ]

    result = publish_clearml_evaluation(
        project_name="MARLIN",
        task_name="oracle-eval-123",
        tags=["evaluation", "oracle"],
        metrics=metrics,
        rows=rows,
        settings={"lane": "dreams"},
    )

    assert result == {
        "task_id": "task-id",
        "task_name": "oracle-eval-123",
        "project_name": "MARLIN",
        "web_url": "https://clearml.example/task-id",
    }
    assert {call["series"]: call["value"] for call in scalar_calls} == {
        "Exact@1": 0.5,
        "Exact@10": 1.0,
        "Formula@1": 0.0,
        "Formula@10": 0.5,
        "Candidate return": 1.0,
        "Validity": 0.75,
        "Mass validity": 0.5,
        "Uniqueness": 0.25,
        "Tanimoto@1 (returned)": 0.4,
    }
    assert table_calls[0]["title"] == "MARLIN molecular evaluation"
    assert table_calls[0]["table_plot"].to_dict("records") == [
        {
            "spec_name": "spectrum-1",
            "rank": 1,
            "target_smiles": "CCO",
            "candidate_smiles": "CCN",
            "exact": False,
            "formula_match": False,
            "target_tanimoto": 0.6,
            "mass_error_ppm": 2.5,
        },
        {
            "spec_name": "spectrum-1",
            "rank": 2,
            "target_smiles": "CCO",
            "candidate_smiles": "CCO",
            "exact": True,
            "formula_match": True,
            "target_tanimoto": 1.0,
            "mass_error_ppm": 0.1,
        },
    ]
    assert connected == [("evaluation_settings", {"lane": "dreams"})]
    assert closed == [True]


def test_publish_clearml_evaluation_attaches_without_editing_completed_task(
    monkeypatch,
):
    scalar_calls = []
    connected = []
    closed = []

    class FakeLogger:
        def report_scalar(self, **kwargs):
            scalar_calls.append(kwargs)

        def report_table(self, **kwargs):
            pass

    attached_task = SimpleNamespace(
        id="completed-task-id",
        name="completed-training",
        get_logger=lambda: FakeLogger(),
        connect=lambda value, name: connected.append((name, value)),
        get_output_log_web_page=lambda: "https://clearml.example/completed-task-id",
        close=lambda: closed.append(True),
    )

    class FakeTask:
        TaskTypes = SimpleNamespace(testing="testing")

        @staticmethod
        def get_task(*, task_id):
            assert task_id == "completed-task-id"
            return attached_task

    monkeypatch.setitem(sys.modules, "clearml", SimpleNamespace(Task=FakeTask))

    result = publish_clearml_evaluation(
        project_name=None,
        task_name=None,
        tags=[],
        metrics={"rows": 1, "validity": 1.0},
        rows=[],
        settings={"lane": "dreams"},
        task_id="completed-task-id",
        iteration=1,
    )

    assert result["task_id"] == "completed-task-id"
    assert scalar_calls[0]["series"] == "Validity"
    assert connected == []
    assert closed == []
