from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_mgf_training_overlap.py"
SPEC = importlib.util.spec_from_file_location("audit_mgf_training_overlap", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_iter_mgf_inchikeys_streams_full_and_missing_keys(tmp_path: Path) -> None:
    mgf_path = tmp_path / "training.mgf"
    mgf_path.write_text(
        "\n".join(
            [
                "BEGIN IONS",
                "INCHIKEY=AAAAAAAAAAAAAA-BBBBBBBBBB-C",
                "100.0 1.0",
                "END IONS",
                "BEGIN IONS",
                "INCHI_AUX=CCCCCCCCCCCCCC-DDDDDDDDDD-E",
                "200.0 1.0",
                "END IONS",
                "BEGIN IONS",
                "300.0 1.0",
                "END IONS",
            ]
        )
    )

    assert list(MODULE.iter_mgf_inchikeys(mgf_path)) == [
        "AAAAAAAAAAAAAA",
        "CCCCCCCCCCCCCC",
        None,
    ]


def test_main_reports_row_and_structure_overlap(tmp_path: Path) -> None:
    mgf_path = tmp_path / "training.mgf"
    mgf_path.write_text(
        "\n".join(
            [
                "BEGIN IONS",
                "INCHIKEY=AAAAAAAAAAAAAA-BBBBBBBBBB-C",
                "100.0 1.0",
                "END IONS",
                "BEGIN IONS",
                "INCHIKEY=AAAAAAAAAAAAAA-XXXXXXXXXX-Y",
                "200.0 1.0",
                "END IONS",
                "BEGIN IONS",
                "INCHIKEY=CCCCCCCCCCCCCC-DDDDDDDDDD-E",
                "300.0 1.0",
                "END IONS",
            ]
        )
    )
    metadata_path = tmp_path / "evaluation.csv"
    pd.DataFrame(
        {
            "spectrum_id": ["one", "one", "two", "three"],
            "inchi_key_first_block": [
                "AAAAAAAAAAAAAA",
                "AAAAAAAAAAAAAA",
                "BBBBBBBBBBBBBB",
                "CCCCCCCCCCCCCC",
            ],
            "benchmark_partition": ["evaluation", "evaluation", "calibration", "evaluation"],
        }
    ).to_csv(metadata_path, index=False)
    output_path = tmp_path / "audit.json"

    assert (
        MODULE.main(
            [
                "--training-mgf",
                str(mgf_path),
                "--evaluation-metadata",
                str(metadata_path),
                "--output",
                str(output_path),
                "--filter-column",
                "benchmark_partition",
                "--filter-value",
                "evaluation",
                "--deduplicate-column",
                "spectrum_id",
            ]
        )
        == 0
    )

    result = json.loads(output_path.read_text())
    assert result["training"]["rows"] == 3
    assert result["training"]["unique_structure_blocks"] == 2
    assert result["evaluation"]["rows"] == 2
    assert result["overlap"]["rows"] == 2
    assert result["overlap"]["unique_structure_blocks"] == 2
    assert result["overlap"]["examples"][0]["training_spectrum_count"] == 2


@pytest.mark.parametrize("value", ["short", "AAAAAAAAAAAAA1", ""])
def test_inchikey_first_block_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="Invalid InChIKey"):
        MODULE.inchikey_first_block(value)
