import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts.evaluate_marlin_nplib1 import (
    _clearml_candidate_table,
    _clearml_decoding_diagnostic_table,
    add_formula_metrics,
    formula_metric_summary,
    parse_args,
    publish_clearml_evaluation,
    validate_evaluation_profile,
)


_MINIMAL_EVALUATION_ARGV = [
    "evaluate_marlin_nplib1.py",
    "--checkpoint", "checkpoint.ckpt",
    "--tokenizer", "tokenizer.json",
    "--metadata", "metadata.csv",
    "--fingerprints", "fingerprints.npz",
    "--fingerprint-key", "probs",
    "--lane", "dreams",
    "--output-dir", "out",
]


def test_mass_reachability_prune_is_opt_in_on_the_evaluation_cli(monkeypatch):
    monkeypatch.setattr(sys, "argv", _MINIMAL_EVALUATION_ARGV)

    assert parse_args().mass_reachability_prune is False


def test_mass_reachability_prune_flag_enables_the_deviation(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", _MINIMAL_EVALUATION_ARGV + ["--mass-reachability-prune"]
    )

    assert parse_args().mass_reachability_prune is True


def test_paper_parity_requires_the_paper_candidate_budget():
    with pytest.raises(ValueError, match="exactly 384"):
        validate_evaluation_profile(
            "paper-parity",
            candidates=16,
            spec_manifest=Path("nplib1_val_micro64_v1.tsv"),
            max_spectra=None,
        )


def test_paper_parity_rejects_locked_test_and_truncated_panels():
    with pytest.raises(ValueError, match="nplib1_val"):
        validate_evaluation_profile(
            "paper-parity",
            candidates=384,
            spec_manifest=Path("nplib1_test_locked_full803_v1.tsv"),
            max_spectra=None,
        )
    with pytest.raises(ValueError, match="cannot truncate"):
        validate_evaluation_profile(
            "paper-parity",
            candidates=384,
            spec_manifest=Path("nplib1_val_micro64_v1.tsv"),
            max_spectra=64,
        )


def test_mass_reachability_prune_reaches_the_grammar_mask_and_the_manifest():
    source = Path("scripts/evaluate_marlin_nplib1.py").read_text()

    assert "mass_reachability_prune=args.mass_reachability_prune" in source
    assert '"mass_reachability_prune": args.mass_reachability_prune' in source


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


def test_clearml_decoding_diagnostic_table_bounds_terminal_samples():
    table = _clearml_decoding_diagnostic_table(
        [
            {
                "spec_name": "spectrum-1",
                "attempts": 16,
                "valid": 1,
                "mass_valid": 0,
                "constraint_dead_ends": 2,
                "eos_terminated": 3,
                "max_length_terminated": 11,
                "sample_terminal_safes": [f"safe-{index}" for index in range(7)],
                "sample_dead_ends": [
                    {"position": index} for index in range(7)
                ],
            }
        ]
    )

    record = table.to_dict("records")[0]
    assert record["sample_terminal_safes"].splitlines() == [
        f"safe-{index}" for index in range(5)
    ]
    assert json.loads(record["sample_dead_ends"]) == [
        {"position": index} for index in range(5)
    ]
    assert record["eos_terminated"] == 3
    assert record["max_length_terminated"] == 11


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
        "eos_terminated_mean": 0.75,
        "max_length_terminated_mean": 15.25,
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
        "EOS terminated": 0.75,
        "Max-length terminated": 15.25,
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
    assert table_calls[1]["title"] == "MARLIN decoding diagnostics"
    assert table_calls[1]["table_plot"].to_dict("records")[0][
        "spec_name"
    ] == "spectrum-1"
    assert connected == [("evaluation_settings", {"lane": "dreams"})]
    assert closed == [True]


def test_publish_clearml_evaluation_uses_separate_profile_namespaces(monkeypatch):
    scalar_calls = []

    class FakeLogger:
        def report_scalar(self, **kwargs):
            scalar_calls.append(kwargs)

        def report_table(self, **kwargs):
            pass

    class FakeTask:
        TaskTypes = SimpleNamespace(testing="testing")

        @staticmethod
        def init(**kwargs):
            return SimpleNamespace(
                id="task-id",
                name=kwargs["task_name"],
                get_logger=lambda: FakeLogger(),
                connect=lambda value, name: None,
                get_output_log_web_page=lambda: "https://clearml.example/task-id",
                close=lambda: None,
            )

    monkeypatch.setitem(sys.modules, "clearml", SimpleNamespace(Task=FakeTask))
    for profile, expected_title in (
        ("screening", "Molecular screening"),
        ("paper-parity", "Paper parity"),
    ):
        scalar_calls.clear()
        publish_clearml_evaluation(
            project_name="MARLIN",
            task_name=f"{profile}-eval",
            tags=[],
            metrics={"rows": 1, "exact_top1": 0.0},
            rows=[],
            settings={},
            evaluation_profile=profile,
        )
        assert scalar_calls[0]["title"] == expected_title


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
