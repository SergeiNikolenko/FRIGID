import json

import numpy as np
import pandas as pd
import pytest

from marlin.evaluation import (
    MIST_FORMULA_SOURCE,
    MIST_LANE_KIND,
    load_fingerprints,
    mass_bin_metrics,
    validate_mist_lane_provenance,
)


def test_load_fingerprints_accepts_leading_metadata_subset_without_ids(tmp_path):
    fingerprints = np.zeros((3, 4096), dtype=np.float32)
    fingerprints[0, 7] = 1.0
    fingerprints[1, 11] = 1.0
    path = tmp_path / "fingerprints.npz"
    np.savez(path, mist_binary=fingerprints)
    metadata = pd.DataFrame({"spec_name": ["spectrum-0"]})

    loaded = load_fingerprints(
        path,
        "mist_binary",
        None,
        metadata,
        allow_leading_subset=True,
    )

    assert loaded.shape == (1, 4096)
    assert np.array_equal(loaded, fingerprints[:1])


def test_load_fingerprints_can_preserve_soft_probabilities(tmp_path):
    fingerprints = np.zeros((1, 4096), dtype=np.float32)
    fingerprints[0, [7, 11]] = [0.61, 0.99]
    path = tmp_path / "fingerprints.npz"
    np.savez(path, probs=fingerprints)
    metadata = pd.DataFrame({"spec_name": ["spectrum-0"]})

    loaded = load_fingerprints(
        path,
        "probs",
        0.5,
        metadata,
        allow_leading_subset=True,
        preserve_probabilities=True,
    )

    assert loaded[0, 7] == pytest.approx(0.61)
    assert loaded[0, 11] == pytest.approx(0.99)


def test_mass_bin_metrics_use_paper_boundaries():
    rows = [
        {"neutral_mass": 299.9, "exact_top1": True},
        {"neutral_mass": 300.0, "exact_top1": False},
        {"neutral_mass": 499.9, "exact_top1": True},
        {"neutral_mass": 500.0, "exact_top1": False},
    ]

    metrics = mass_bin_metrics(rows)

    assert metrics["lt_300"] == {"rows": 1, "exact_top1": 1.0}
    assert metrics["300_to_500"] == {"rows": 2, "exact_top1": 0.5}
    assert metrics["gte_500"] == {"rows": 1, "exact_top1": 0.0}


def _mist_payload(
    fingerprint_path,
    metadata_path,
    formula_manifest_path,
    feature_bridge_manifest_path,
    mist_labels_path,
):
    from marlin.evaluation import sha256_file

    return {
        "kind": MIST_LANE_KIND,
        "formula_source": MIST_FORMULA_SOURCE,
        "rows": 803,
        "fingerprint_bits": 4096,
        "output_sha256": sha256_file(fingerprint_path),
        "reference_metadata_sha256": sha256_file(metadata_path),
        "formula_manifest_sha256": sha256_file(formula_manifest_path),
        "feature_bridge_manifest_sha256": sha256_file(feature_bridge_manifest_path),
        "mist_labels_sha256": sha256_file(mist_labels_path),
        "official_mist_git_commit": "mist-commit",
        "mist_cf_git_commit": "mist-cf-commit",
        "mist_cf_checkpoint_sha256": "mist-cf-checkpoint-sha256",
        "mist_checkpoint_sha256": "checkpoint-sha256",
    }


def test_mist_lane_provenance_rejects_oracle_formula_source(tmp_path):
    fingerprints = tmp_path / "fingerprints.npz"
    metadata = tmp_path / "metadata.csv"
    fingerprints.write_bytes(b"fingerprints")
    metadata.write_text("spec_name\n")
    formula_manifest = tmp_path / "formula.json"
    formula_manifest.write_text("{}")
    feature_bridge_manifest = tmp_path / "feature.json"
    feature_bridge_manifest.write_text("{}")
    mist_labels = tmp_path / "mist_labels.tsv"
    mist_labels.write_text("spec\n")
    path = tmp_path / "manifest.json"
    payload = _mist_payload(
        fingerprints, metadata, formula_manifest, feature_bridge_manifest, mist_labels
    )
    payload["formula_source"] = "ground-truth formula"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="formula-blind adapted lane"):
        validate_mist_lane_provenance(
            path,
            803,
            fingerprints,
            metadata,
            formula_manifest,
            feature_bridge_manifest,
            mist_labels,
        )


def test_mist_lane_provenance_accepts_clean_official_lane(tmp_path):
    fingerprints = tmp_path / "fingerprints.npz"
    metadata = tmp_path / "metadata.csv"
    fingerprints.write_bytes(b"fingerprints")
    metadata.write_text("spec_name\n")
    formula_manifest = tmp_path / "formula.json"
    formula_manifest.write_text("{}")
    feature_bridge_manifest = tmp_path / "feature.json"
    feature_bridge_manifest.write_text("{}")
    mist_labels = tmp_path / "mist_labels.tsv"
    mist_labels.write_text("spec\n")
    path = tmp_path / "manifest.json"
    payload = _mist_payload(
        fingerprints, metadata, formula_manifest, feature_bridge_manifest, mist_labels
    )
    path.write_text(json.dumps(payload))

    assert (
        validate_mist_lane_provenance(
            path,
            803,
            fingerprints,
            metadata,
            formula_manifest,
            feature_bridge_manifest,
            mist_labels,
        )
        == payload
    )


def test_mist_lane_provenance_rejects_fingerprint_hash_mismatch(tmp_path):
    fingerprints = tmp_path / "fingerprints.npz"
    metadata = tmp_path / "metadata.csv"
    fingerprints.write_bytes(b"fingerprints")
    metadata.write_text("spec_name\n")
    formula_manifest = tmp_path / "formula.json"
    formula_manifest.write_text("{}")
    feature_bridge_manifest = tmp_path / "feature.json"
    feature_bridge_manifest.write_text("{}")
    mist_labels = tmp_path / "mist_labels.tsv"
    mist_labels.write_text("spec\n")
    path = tmp_path / "manifest.json"
    payload = _mist_payload(
        fingerprints, metadata, formula_manifest, feature_bridge_manifest, mist_labels
    )
    payload["output_sha256"] = "wrong"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="formula-blind adapted lane"):
        validate_mist_lane_provenance(
            path,
            803,
            fingerprints,
            metadata,
            formula_manifest,
            feature_bridge_manifest,
            mist_labels,
        )


def _row(spec_name, *, returned, tanimoto=0.0, mass=350.0, formula=True):
    row = {
        "spec_name": spec_name,
        "lane": "dreams",
        "neutral_mass": mass,
        "runtime_seconds": 2.0,
        "attempts": 8,
        "truncated": False,
        "constraint_dead_ends": 5,
        "eos_terminated": 1,
        "max_length_terminated": 0,
        "validity": 0.25,
        "completed_validity": 0.5,
        "mass_validity": 0.1,
        "uniqueness": 0.2,
        "candidate_returned": returned,
        "exact_top1": False,
        "exact_top10": False,
        "tanimoto_top1": tanimoto,
        "tanimoto_top10": tanimoto,
    }
    if formula:
        row["formula_top1"] = returned
        row["formula_top10"] = returned
    return row


def test_aggregate_prediction_metrics_keeps_each_metric_denominator():
    """Shard metrics cannot be averaged: Tanimoto is over the spectra that
    returned a candidate while candidate return is over every spectrum."""
    from marlin.evaluation import aggregate_prediction_metrics

    rows = [
        _row("a", returned=True, tanimoto=0.6),
        _row("b", returned=False),
        _row("c", returned=False),
        _row("d", returned=True, tanimoto=0.2, mass=250.0),
    ]

    metrics = aggregate_prediction_metrics(rows)

    assert metrics["rows"] == 4
    assert metrics["candidate_return_rate"] == pytest.approx(0.5)
    assert metrics["tanimoto_top1"] == pytest.approx(0.4)
    assert metrics["formula_top1_all"] == pytest.approx(0.5)
    assert metrics["formula_top1_returned"] == pytest.approx(1.0)
    assert metrics["validity"] == pytest.approx(0.25)
    assert metrics["attempts_total"] == 32
    assert metrics["runtime_seconds_total"] == pytest.approx(8.0)
    assert metrics["mass_bins"]["lt_300"]["rows"] == 1
    assert metrics["lane"] == "dreams"
    # internal_diversity needs the candidate molecules re-fingerprinted, so it is
    # absent rather than silently redefined on the merged path.
    assert "internal_diversity" not in metrics


def test_aggregate_prediction_metrics_refuses_mixed_lanes_and_missing_formula():
    from marlin.evaluation import aggregate_prediction_metrics

    mist = _row("b", returned=True)
    mist["lane"] = "mist"
    with pytest.raises(ValueError, match="mix lanes"):
        aggregate_prediction_metrics([_row("a", returned=True), mist])

    partial = aggregate_prediction_metrics(
        [_row("a", returned=True), _row("b", returned=True, formula=False)]
    )
    assert "formula_top1_all" not in partial
