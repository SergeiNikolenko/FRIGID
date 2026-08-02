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
