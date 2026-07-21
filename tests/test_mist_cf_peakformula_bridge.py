import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.prepare_mist_cf_peakformula_dataset import (
    FEATURE_BRIDGE_KIND,
    FORMULA_MANIFEST_KIND,
    FORMULA_SOURCE,
    build_peakformula_dataset,
)


def _write_inputs(
    tmp_path: Path, *, second_candidate: bool = False
) -> tuple[Path, Path, Path]:
    mgf = tmp_path / "test.mgf"
    mgf.write_text(
        "BEGIN IONS\nSCANS=a\nPEPMASS=43.05422664\n10 1\nEND IONS\n"
        "BEGIN IONS\nSCANS=b\nPEPMASS=57.06987670\n20 2\nEND IONS\n"
    )
    predictions = tmp_path / "predictions.tsv"
    a_rows = (
        "a\tC2H4\t0.9\t[M+H]+\t29.03857658\na\tC3H6\t0.8\t[M+H]+\t43.05422664\n"
        if second_candidate
        else "a\tC3H6\t0.9\t[M+H]+\t43.05422664\n"
    )
    predictions.write_text(
        "spec\tcand_form\tscores\tcand_ion\tparentmasses\n"
        f"{a_rows}"
        "b\tC4H8\t0.8\t[M+H]+\t57.06987670\n"
    )
    subforms = tmp_path / "subform_assigns"
    subforms.mkdir()
    (subforms / "a.json").write_text(
        json.dumps(
            {
                "C3H6": {
                    "cand_ion": "[M+H]+",
                    "cand_tbl": {
                        "formula": ["CH2", "C2H4"],
                        "mz": [15.0230, 29.0387],
                        "ms2_inten": [0.25, 1.0],
                        "mass_diff": [4.0, 5.0],
                        "ions": ["[M+H]+", "[M+H]+"],
                    },
                }
            }
        )
    )
    (subforms / "b.json").write_text(
        json.dumps({"C4H8": {"cand_ion": "[M+H]+", "cand_tbl": None}})
    )
    return mgf, predictions, subforms


def test_build_peakformula_dataset_is_top1_locked_and_records_root_only(
    tmp_path: Path,
) -> None:
    mgf, predictions, subforms = _write_inputs(tmp_path)
    output = tmp_path / "dataset"

    manifest = build_peakformula_dataset(
        mgf,
        predictions,
        subforms,
        output,
        "mist-cf-commit",
        "mist-cf-checkpoint",
    )

    formula_manifest = json.loads((output / "formula_bridge_manifest.json").read_text())
    assert formula_manifest["kind"] == FORMULA_MANIFEST_KIND
    assert formula_manifest["formula_source"] == FORMULA_SOURCE
    assert formula_manifest["fallback_rows"] == 0
    assert formula_manifest["maximum_candidate_rank"] == 1
    assert manifest["kind"] == FEATURE_BRIDGE_KIND
    assert manifest["rows"] == 2
    assert manifest["root_only_rows"] == 1
    assert manifest["root_only_ids"] == ["b"]
    assert manifest["mist_cf_git_commit"] == "mist-cf-commit"
    assert manifest["maximum_subformula_assignment_ppm"] == 5.0

    tree_a = json.loads((output / "peakformula_trees/a.json").read_text())
    tree_b = json.loads((output / "peakformula_trees/b.json").read_text())
    assert [fragment["molecularFormula"] for fragment in tree_a["fragments"]] == [
        "C3H6",
        "CH2",
        "C2H4",
    ]
    assert tree_b["fragments"] == [
        {
            "id": 0,
            "molecularFormula": "C4H8",
            "relativeIntensity": 0.0,
            "mz": pytest.approx(57.0698767),
        }
    ]
    summary = pd.read_csv(
        output / "sirius_outputs/summary_statistics/summary_df.tsv", sep="\t"
    )
    assert summary["spec_name"].tolist() == ["a", "b"]
    with (output / "mist_labels.tsv").open(newline="") as handle:
        assert [row["spec"] for row in csv.DictReader(handle, delimiter="\t")] == [
            "a",
            "b",
        ]


def test_build_peakformula_dataset_rejects_non_top1_mass_fallback(
    tmp_path: Path,
) -> None:
    mgf, predictions, subforms = _write_inputs(tmp_path, second_candidate=True)

    with pytest.raises(ValueError, match="top-1 candidate"):
        build_peakformula_dataset(
            mgf,
            predictions,
            subforms,
            tmp_path / "dataset",
            "mist-cf-commit",
            "mist-cf-checkpoint",
        )


def test_build_peakformula_dataset_rejects_non_subformula(tmp_path: Path) -> None:
    mgf, predictions, subforms = _write_inputs(tmp_path)
    payload = json.loads((subforms / "a.json").read_text())
    payload["C3H6"]["cand_tbl"]["formula"] = ["C4H8", "C2H4"]
    (subforms / "a.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="not a root subformula"):
        build_peakformula_dataset(
            mgf,
            predictions,
            subforms,
            tmp_path / "dataset",
            "mist-cf-commit",
            "mist-cf-checkpoint",
        )


def test_build_peakformula_dataset_treats_empty_table_as_root_only(
    tmp_path: Path,
) -> None:
    mgf, predictions, subforms = _write_inputs(tmp_path)
    payload = json.loads((subforms / "a.json").read_text())
    payload["C3H6"]["cand_tbl"] = {
        "formula": [],
        "mz": [],
        "ms2_inten": [],
        "mass_diff": [],
        "ions": [],
    }
    (subforms / "a.json").write_text(json.dumps(payload))

    manifest = build_peakformula_dataset(
        mgf,
        predictions,
        subforms,
        tmp_path / "dataset",
        "mist-cf-commit",
        "mist-cf-checkpoint",
    )

    assert manifest["root_only_rows"] == 2
    assert manifest["root_only_ids"] == ["a", "b"]
