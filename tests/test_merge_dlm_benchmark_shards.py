from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "merge_dlm_benchmark_shards.py"
SPEC = importlib.util.spec_from_file_location("merge_dlm_benchmark_shards", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["merge_dlm_benchmark_shards"] = MODULE
SPEC.loader.exec_module(MODULE)


FILES = (
    "predictions_mist_binary.csv",
    "prediction_scores_mist_binary.csv",
    "detailed_results.csv",
)


def _write_shard(path, names, start):
    path.mkdir()
    (path / "RUN_MANIFEST.json").write_text(json.dumps({"start": start}))
    for filename in FILES:
        column = "name" if filename == "predictions_mist_binary.csv" else "spec_name"
        pd.DataFrame({column: names, "value": range(len(names))}).to_csv(
            path / filename, index=False
        )


def _write_manifest(path, names):
    pd.DataFrame({"spec_name": names}).to_csv(path, sep="\t", index=False)


def test_merges_disjoint_shards_in_manifest_order(tmp_path):
    expected = tmp_path / "expected.tsv"
    names = ["q3", "q1", "q2"]
    _write_manifest(expected, names)
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_shard(first, ["q1"], 0)
    _write_shard(second, ["q2", "q3"], 1)

    output = tmp_path / "merged"
    manifest = MODULE.merge_shards([first, second], expected, output)

    assert manifest["coverage"] == {"expected": 3, "observed": 3, "missing": [], "extra": []}
    assert pd.read_csv(output / "prediction_scores_mist_binary.csv")["spec_name"].tolist() == names


def test_rejects_overlapping_shards(tmp_path):
    expected = tmp_path / "expected.tsv"
    _write_manifest(expected, ["q1", "q2"])
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_shard(first, ["q1"], 0)
    _write_shard(second, ["q1", "q2"], 1)

    with pytest.raises(ValueError, match="Overlapping shard queries"):
        MODULE.merge_shards([first, second], expected, tmp_path / "merged")
