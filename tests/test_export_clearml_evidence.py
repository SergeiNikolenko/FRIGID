from scripts.export_clearml_evidence import scalar_evidence


def test_scalar_evidence_preserves_series_and_counts() -> None:
    reported = {
        "train": {
            "loss": {"x": [0, 1], "y": [2.0, 1.0]},
        }
    }

    assert scalar_evidence(reported) == {
        "train": {
            "loss": {
                "count": 2,
                "x": [0, 1],
                "y": [2.0, 1.0],
            }
        }
    }
