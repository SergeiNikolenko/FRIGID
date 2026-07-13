from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "merge_mces_shards.py"
SPEC = importlib.util.spec_from_file_location("merge_mces_shards", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["merge_mces_shards"] = MODULE
SPEC.loader.exec_module(MODULE)


def write_expected(path: Path, names: list[str]) -> None:
    pd.DataFrame(
        {
            "spec_name": names,
            "inchikey_first_block": [f"CLUSTER{index}" for index in range(len(names))],
        }
    ).to_csv(path, sep="\t", index=False)


def write_shard(
    path: Path,
    expected_manifest: Path,
    names: list[str],
    start: int,
    *,
    prediction_sha: str = "c" * 64,
) -> None:
    path.mkdir()
    rows = []
    for index, name in enumerate(names):
        for variant in ("control", "union"):
            offset = -1.0 if variant == "union" else 0.0
            row = {"spec_name": name, "variant": variant}
            row.update(
                {column: float(20 + index - rank + offset) for rank, column in enumerate(MODULE.MCES_COLUMNS)}
            )
            rows.append(row)
    output_path = path / "per_sample_mces.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    manifest = {
        "schema_version": 1,
        "purpose": "frigid_full_mces_shard",
        "status": "completed",
        "exit_code": 0,
        "code": {"commit": "f" * 40, "dirty": False},
        "selection": {
            "start_index": start,
            "max_spectra": len(names),
            "expected_total": 4,
        },
        "inputs": {
            "expected_manifest": {
                "path": str(expected_manifest),
                "sha256": MODULE.sha256_file(expected_manifest),
            },
            "runtime_manifest": {"path": "/runtime.json", "sha256": "a" * 64},
            "predictions": [
                {"name": "control", "path": "/control.csv", "sha256": "b" * 64},
                {"name": "union", "path": "/union.csv", "sha256": prediction_sha},
            ],
        },
        "runtime": {"versions": {"PuLP": "2.7.0", "myopic-mces": "1.0.1"}},
        "settings": {
            "variant_order": ["control", "union"],
            "top_k": 10,
            "threshold": 15,
            "n_jobs": 4,
        },
        "progress": {
            "processed_spectra": len(names),
            "expected_spectra": len(names),
        },
        "outputs": {
            "per_sample_mces": {
                "path": str(output_path),
                "sha256": MODULE.sha256_file(output_path),
                "row_count": len(rows),
            }
        },
    }
    (path / "RUN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_merges_exact_coverage_and_reports_lower_is_better(tmp_path: Path) -> None:
    expected = tmp_path / "expected.tsv"
    names = ["q3", "q1", "q4", "q2"]
    write_expected(expected, names)
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_shard(first, expected, names[:2], 0)
    write_shard(second, expected, names[2:], 2)

    output = tmp_path / "merged"
    manifest = MODULE.merge_shards(
        [second, first], expected, output, bootstrap_resamples=100, seed=7
    )

    merged = pd.read_csv(output / "per_sample_mces.csv")
    assert list(zip(merged["spec_name"], merged["variant"])) == [
        (name, variant) for name in names for variant in ("control", "union")
    ]
    assert manifest["coverage"] == {
        "expected_queries": 4,
        "observed_queries": 4,
        "observed_rows": 8,
        "missing_indices": [],
    }
    summary = json.loads(
        (output / "paired_control_vs_union" / "comparison_summary.json").read_text()
    )
    assert summary["lower_is_better"] is True
    assert summary["metrics"]["mces@1"]["mean_delta"] == -1.0
    assert summary["metrics"]["mces@1"]["improved"] == 4
    aggregate = json.loads((output / "aggregate_statistics.json").read_text())
    assert aggregate["query_count"] == 4
    assert set(aggregate["variants"]) == {"control", "union"}


def test_rejects_overlapping_ranges(tmp_path: Path) -> None:
    expected = tmp_path / "expected.tsv"
    names = ["q1", "q2", "q3", "q4"]
    write_expected(expected, names)
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_shard(first, expected, names[:2], 0)
    write_shard(second, expected, names[:2], 0)

    with pytest.raises(ValueError, match="Overlapping MCES shard range"):
        MODULE.merge_shards([first, second], expected, tmp_path / "merged")


def test_rejects_frozen_provenance_drift(tmp_path: Path) -> None:
    expected = tmp_path / "expected.tsv"
    names = ["q1", "q2", "q3", "q4"]
    write_expected(expected, names)
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_shard(first, expected, names[:2], 0)
    write_shard(second, expected, names[2:], 2, prediction_sha="d" * 64)

    with pytest.raises(ValueError, match="provenance/settings mismatch"):
        MODULE.merge_shards([first, second], expected, tmp_path / "merged")


def test_rejects_modified_output(tmp_path: Path) -> None:
    expected = tmp_path / "expected.tsv"
    names = ["q1", "q2", "q3", "q4"]
    write_expected(expected, names)
    shard = tmp_path / "shard"
    write_shard(shard, expected, names, 0)
    with (shard / "per_sample_mces.csv").open("a", encoding="utf-8") as handle:
        handle.write("corrupt\n")

    with pytest.raises(ValueError, match="output SHA-256 mismatch"):
        MODULE.merge_shards([shard], expected, tmp_path / "merged")
