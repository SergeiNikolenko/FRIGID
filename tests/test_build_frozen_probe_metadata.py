from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_frozen_probe_metadata.py"
SPEC = importlib.util.spec_from_file_location("build_frozen_probe_metadata", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_metadata_matches_mist_atom_filter(tmp_path):
    labels_path = tmp_path / "labels.tsv"
    split_path = tmp_path / "split.tsv"
    pd.DataFrame(
        {
            "spec": ["a", "b", "c"],
            "smiles": ["CCO", "C[Br]", "CCN"],
            "inchikey": ["AAAA", "BBBB", "CCCC-DD"],
            "formula": ["C2H6O", "CH3Br", "C2H7N"],
        }
    ).to_csv(labels_path, sep="\t", index=False)
    pd.DataFrame(
        {"name": ["a", "b", "c"], "split": ["train", "train", "val"]}
    ).to_csv(split_path, sep="\t", index=False)

    metadata, summary = MODULE.build_metadata(
        labels_path, split_path, split="train"
    )

    assert metadata["spec_name"].tolist() == ["a"]
    assert metadata["fingerprint_index"].tolist() == [0]
    assert metadata["inchi_key_first_block"].tolist() == ["AAAA"]
    assert summary["source_rows"] == 2
    assert summary["eligible_rows"] == 1
    assert summary["unsupported_atom_counts"] == {"Br": 1}


def test_build_metadata_preserves_label_order(tmp_path):
    labels_path = tmp_path / "labels.tsv"
    split_path = tmp_path / "split.tsv"
    pd.DataFrame(
        {
            "spec": ["b", "a"],
            "smiles": ["CCO", "CCN"],
            "inchikey": ["BBBB", "AAAA"],
        }
    ).to_csv(labels_path, sep="\t", index=False)
    pd.DataFrame(
        {"name": ["a", "b"], "split": ["train", "train"]}
    ).to_csv(split_path, sep="\t", index=False)

    metadata, _ = MODULE.build_metadata(labels_path, split_path, split="train")

    assert metadata["spec_name"].tolist() == ["b", "a"]
