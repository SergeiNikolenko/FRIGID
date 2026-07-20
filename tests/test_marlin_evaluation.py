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


def test_mist_lane_provenance_rejects_oracle_formula_source(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "kind": MIST_LANE_KIND,
                "formula_source": "ground-truth formula",
                "rows": 803,
                "fingerprint_bits": 4096,
            }
        )
    )

    with pytest.raises(ValueError, match="formula-blind official lane"):
        validate_mist_lane_provenance(path, 803)


def test_mist_lane_provenance_accepts_clean_official_lane(tmp_path):
    path = tmp_path / "manifest.json"
    payload = {
        "kind": MIST_LANE_KIND,
        "formula_source": MIST_FORMULA_SOURCE,
        "rows": 803,
        "fingerprint_bits": 4096,
    }
    path.write_text(json.dumps(payload))

    assert validate_mist_lane_provenance(path, 803) == payload
