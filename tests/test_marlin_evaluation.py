import numpy as np
import pandas as pd

from marlin.evaluation import load_fingerprints


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
