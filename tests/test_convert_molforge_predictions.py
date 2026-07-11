import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "convert_molforge_predictions.py"
SPEC = importlib.util.spec_from_file_location("convert_molforge_predictions", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_convert_preserves_manifest_order_and_candidate_ranks(tmp_path):
    manifest = tmp_path / "subset.tsv"
    manifest.write_text("spec_name\nq2\nq1\n")
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions,
        [
            {
                "spec_name": "q2",
                "true_smiles": "target-must-not-be-copied",
                "threshold": 0.172,
                "pred_smiles_top10": ["CC", "CCC"],
            },
            {
                "spec_name": "q1",
                "threshold": 0.172,
                "pred_smiles_top10": ["CO"],
            },
        ],
    )
    output = tmp_path / "candidates.csv"

    query_count, candidate_count = MODULE.convert_predictions(
        predictions, output, manifest
    )

    assert (query_count, candidate_count) == (2, 3)
    assert output.read_text().splitlines() == [
        "query_spec_name,rank,candidate_smiles,source_threshold",
        "q2,1,CC,0.172",
        "q2,2,CCC,0.172",
        "q1,1,CO,0.172",
    ]
    run_manifest = json.loads(output.with_suffix(".csv.manifest.json").read_text())
    assert run_manifest["target_fields_used"] == []


def test_target_fields_cannot_change_candidate_csv(tmp_path):
    manifest = tmp_path / "subset.tsv"
    manifest.write_text("spec_name\nq1\n")
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    common = {"spec_name": "q1", "threshold": 0.5, "pred_smiles_top10": ["CC"]}
    write_jsonl(first, [{**common, "true_smiles": "C", "true_formula": "CH4"}])
    write_jsonl(second, [{**common, "true_smiles": "NNN", "true_formula": "N3"}])

    output_first = tmp_path / "first.csv"
    output_second = tmp_path / "second.csv"
    MODULE.convert_predictions(first, output_first, manifest)
    MODULE.convert_predictions(second, output_second, manifest)

    assert output_first.read_bytes() == output_second.read_bytes()


def test_rejects_prediction_order_mismatch(tmp_path):
    manifest = tmp_path / "subset.tsv"
    manifest.write_text("spec_name\nq1\nq2\n")
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(
        predictions,
        [
            {"spec_name": "q2", "pred_smiles_top10": ["CC"]},
            {"spec_name": "q1", "pred_smiles_top10": ["CO"]},
        ],
    )

    with pytest.raises(ValueError, match="manifest order"):
        MODULE.convert_predictions(predictions, tmp_path / "out.csv", manifest)


def test_rejects_empty_candidate_list(tmp_path):
    manifest = tmp_path / "subset.tsv"
    manifest.write_text("spec_name\nq1\n")
    predictions = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, [{"spec_name": "q1", "pred_smiles_top10": []}])

    with pytest.raises(ValueError, match="no candidates"):
        MODULE.convert_predictions(predictions, tmp_path / "out.csv", manifest)
