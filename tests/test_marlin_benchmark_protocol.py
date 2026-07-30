import json
from pathlib import Path

import pandas as pd
import pytest

from marlin.benchmark_selection import (
    hash_spec_names,
    load_spec_manifest,
    select_metadata,
)
from scripts.compare_marlin_benchmark_runs import (
    compare_runs,
    validate_signatures,
)


def test_manifest_selection_preserves_declared_order(tmp_path: Path) -> None:
    manifest = tmp_path / "panel.tsv"
    manifest.write_text("spec_name\tpanel\nc\tmicro\na\tmicro\n")
    metadata = pd.DataFrame({"spec_name": ["a", "b", "c"], "value": [1, 2, 3]})

    names = load_spec_manifest(manifest)
    selected = select_metadata(metadata, names, None)

    assert selected["spec_name"].tolist() == ["c", "a"]
    assert selected["value"].tolist() == [3, 1]
    assert hash_spec_names(names) == hash_spec_names(["c", "a"])


def test_manifest_cannot_be_silently_truncated(tmp_path: Path) -> None:
    manifest = tmp_path / "panel.csv"
    manifest.write_text("spec_name\na\nb\n")
    metadata = pd.DataFrame({"spec_name": ["a", "b"]})

    with pytest.raises(ValueError, match="cannot truncate"):
        select_metadata(metadata, load_spec_manifest(manifest), 1)


def _prediction_frame(exact: tuple[bool, bool]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "spec_name": "a",
                "target_inchikey_first_block": "MOL-A",
                "target_smiles": "CC",
                "neutral_mass": 30.0,
                "lane": "dreams",
                "exact_top1": exact[0],
                "exact_top10": exact[0],
            },
            {
                "spec_name": "b",
                "target_inchikey_first_block": "MOL-B",
                "target_smiles": "CO",
                "neutral_mass": 32.0,
                "lane": "dreams",
                "exact_top1": exact[1],
                "exact_top10": exact[1],
            },
        ]
    )


def test_paired_comparison_reports_exact_delta_and_cluster_ci() -> None:
    summary, paired = compare_runs(
        _prediction_frame((False, False)),
        _prediction_frame((True, False)),
        ["exact_top1"],
        resamples=200,
        confidence=0.95,
        seed=7,
    )

    metric = summary["metrics"]["exact_top1"]
    assert metric["reference_mean"] == 0.0
    assert metric["candidate_mean"] == 0.5
    assert metric["mean_delta"] == 0.5
    assert metric["wins"] == 1
    assert paired["exact_top1_delta"].tolist() == [1.0, 0.0]


def test_signature_contract_rejects_changed_generation_budget(
    tmp_path: Path,
) -> None:
    settings = {
        "lane": "dreams",
        "candidates": 16,
        "seed": 42,
        "ordered_spec_names_sha256": hash_spec_names(["a"]),
    }
    for name, candidates in (("reference", 16), ("candidate", 32)):
        run = tmp_path / name
        run.mkdir()
        current = {**settings, "candidates": candidates}
        (run / "run_signature.json").write_text(
            json.dumps({"git_commit": name, "settings": current})
        )

    with pytest.raises(ValueError, match="settings mismatch"):
        validate_signatures(tmp_path / "reference", tmp_path / "candidate")
