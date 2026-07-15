from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "enrich_encoder_benchmark_metadata.py"
SPEC = importlib.util.spec_from_file_location(
    "enrich_encoder_benchmark_metadata", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_enrich_metadata_preserves_reference_order():
    reference = pd.DataFrame(
        {"fingerprint_index": [0, 1], "spec_name": ["b", "a"]}
    )
    labels = pd.DataFrame(
        {
            "spec": ["a", "b"],
            "ionization": ["[M+Na]+", "[M+H]+"],
            "instrument": ["QTOF", "Orbitrap"],
        }
    )

    enriched = MODULE.enrich_metadata(
        reference,
        labels,
        reference_id_column="spec_name",
        labels_id_column="spec",
    )

    assert enriched["spec_name"].tolist() == ["b", "a"]
    assert enriched["ionization"].tolist() == ["[M+H]+", "[M+Na]+"]
    assert enriched["instrument"].tolist() == ["Orbitrap", "QTOF"]


def test_enrich_metadata_rejects_missing_labels():
    reference = pd.DataFrame({"spec_name": ["a", "missing"]})
    labels = pd.DataFrame({"spec": ["a"], "ionization": ["[M+H]+"]})

    with pytest.raises(ValueError, match="missing label fields"):
        MODULE.enrich_metadata(
            reference,
            labels,
            reference_id_column="spec_name",
            labels_id_column="spec",
        )
