from __future__ import annotations

import importlib.util
import json
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
    "fingerprint_drift.csv",
    "paired_comparison.csv",
)


def _write_manifest(path: Path, names: list[str]) -> None:
    pd.DataFrame({"spec_name": names}).to_csv(path, sep="\t", index=False)


def _write_shard(
    path: Path,
    names: list[str],
    start: int,
    expected_manifest: Path,
    *,
    status: str = "completed",
    softmax_temp: float = 1.0,
) -> None:
    path.mkdir()
    metric_values = {
        "exact_match_top1": 0.0,
        "exact_match_top10": 0.0,
        "tanimoto_top1": 0.5,
        "tanimoto_top10": 0.6,
        "mist_tanimoto": 0.7,
        "total_formula_matched": 2,
        "formula_matches_collected": 3,
        "total_generated": 10,
        "generation_time": 1.5,
    }
    for filename in FILES:
        column = "name" if filename == "predictions_mist_binary.csv" else "spec_name"
        data: dict[str, object] = {column: names, "value": range(len(names))}
        if filename == "detailed_results.csv":
            data.update({key: [value] * len(names) for key, value in metric_values.items()})
        pd.DataFrame(data).to_csv(path / filename, index=False)

    constant_hash = "a" * 64
    inputs = {
        name: {"path": f"/{name}", "sha256": constant_hash}
        for name in MODULE.INPUTS
    }
    inputs["expected_manifest"]["sha256"] = MODULE.sha256_file(expected_manifest)
    outputs = {
        filename: {"sha256": MODULE.sha256_file(path / filename)}
        for filename in FILES
    }
    manifest = {
        "schema_version": 2,
        "purpose": "frigid_msg_full_dlm_shard",
        "status": status,
        "source_name": "dlm_control_no_ngboost100",
        "code": {"commit": "f" * 40, "dirty": False},
        "inputs": inputs,
        "selection": {
            "split": "test",
            "start_index": start,
            "max_spectra": len(names),
        },
        "settings": {
            "softmax_temp": softmax_temp,
            "max_attempts": 100,
            "seed": 42,
        },
        "outputs": outputs,
    }
    (path / "RUN_MANIFEST.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def test_merges_disjoint_shards_in_manifest_order(tmp_path: Path) -> None:
    expected = tmp_path / "expected.tsv"
    names = ["q3", "q1", "q2"]
    _write_manifest(expected, names)
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_shard(first, ["q1", "q2"], 1, expected)
    _write_shard(second, ["q3"], 0, expected)

    output = tmp_path / "merged"
    manifest = MODULE.merge_shards([first, second], expected, output)

    assert manifest["coverage"] == {
        "expected": 3,
        "observed": 3,
        "missing": [],
        "extra": [],
    }
    assert (
        pd.read_csv(output / "prediction_scores_mist_binary.csv")["spec_name"].tolist()
        == names
    )
    aggregate = json.loads((output / "aggregate_statistics.json").read_text())
    assert aggregate["total_spectra"] == 3
    assert aggregate["tanimoto_top1_mean"] == 0.5


def test_rejects_overlapping_shards(tmp_path: Path) -> None:
    expected = tmp_path / "expected.tsv"
    _write_manifest(expected, ["q1", "q2"])
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_shard(first, ["q1"], 0, expected)
    _write_shard(second, ["q1"], 0, expected)

    with pytest.raises(ValueError, match="Overlapping shard queries"):
        MODULE.merge_shards([first, second], expected, tmp_path / "merged")


def test_rejects_frozen_setting_mismatch(tmp_path: Path) -> None:
    expected = tmp_path / "expected.tsv"
    _write_manifest(expected, ["q1", "q2"])
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_shard(first, ["q1"], 0, expected)
    _write_shard(second, ["q2"], 1, expected, softmax_temp=0.8)

    with pytest.raises(ValueError, match="provenance/settings mismatch"):
        MODULE.merge_shards([first, second], expected, tmp_path / "merged")


def test_rejects_incomplete_shard(tmp_path: Path) -> None:
    expected = tmp_path / "expected.tsv"
    _write_manifest(expected, ["q1"])
    shard = tmp_path / "shard"
    _write_shard(shard, ["q1"], 0, expected, status="running")

    with pytest.raises(ValueError, match="Shard is not completed"):
        MODULE.merge_shards([shard], expected, tmp_path / "merged")


def test_rejects_modified_shard_output(tmp_path: Path) -> None:
    expected = tmp_path / "expected.tsv"
    _write_manifest(expected, ["q1"])
    shard = tmp_path / "shard"
    _write_shard(shard, ["q1"], 0, expected)
    with (shard / "detailed_results.csv").open("a", encoding="utf-8") as handle:
        handle.write("corrupt\n")

    with pytest.raises(ValueError, match="output SHA-256 mismatch"):
        MODULE.merge_shards([shard], expected, tmp_path / "merged")
