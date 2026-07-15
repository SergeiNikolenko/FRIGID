import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "benchmark_encoder_predictions.py"
SPEC = importlib.util.spec_from_file_location("benchmark_encoder_predictions", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_cli_writes_ranked_and_per_spectrum_outputs(tmp_path: Path):
    metadata_path = tmp_path / "metadata.csv"
    fingerprints_path = tmp_path / "fingerprints.npz"
    candidate_path = tmp_path / "candidate.npz"
    training_path = tmp_path / "candidate_train.txt"
    output_dir = tmp_path / "results"

    pd.DataFrame(
        {
            "fingerprint_index": [0, 1],
            "spec_name": ["spec-a", "spec-b"],
            "inchi_key_first_block": ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"],
            "adduct": ["[M+H]+", "[M-H]-"],
        }
    ).to_csv(metadata_path, index=False)
    targets = np.asarray([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=np.float32)
    np.savez_compressed(
        fingerprints_path,
        ground_truth=targets,
        mist_probs=np.asarray([[0.9, 0.6, 0.8, 0.2], [0.6, 0.7, 0.1, 0.2]]),
    )
    np.savez_compressed(
        candidate_path,
        spectrum_ids=np.asarray(["spec-b", "spec-a"]),
        probs=np.asarray([[0.1, 0.9, 0.1, 0.1], [0.9, 0.1, 0.8, 0.2]]),
        inference_seconds=np.asarray([2.0, 1.0]),
    )
    training_path.write_text("CCCCCCCCCCCCCC\n")

    exit_code = MODULE.main(
        [
            "--reference-metadata",
            str(metadata_path),
            "--reference-fingerprints",
            str(fingerprints_path),
            "--reference-model",
            "mist=mist_probs",
            "--prediction",
            f"candidate={candidate_path}",
            "--threshold",
            "mist=0.5",
            "--threshold",
            "candidate=0.5",
            "--baseline",
            "mist",
            "--minimum-gain",
            "0.005",
            "--bootstrap-samples",
            "20",
            "--expected-bits",
            "4",
            "--training-identifiers",
            f"candidate={training_path}",
            "--stratify-column",
            "adduct",
            "--code-revision",
            "test-revision",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "benchmark_summary.json").read_text())
    assert summary["baseline"] == "mist"
    assert summary["benchmark_code_revision"] == "test-revision"
    assert [row["model"] for row in summary["ranking"]] == ["candidate", "mist"]
    assert summary["ranking"][0]["promotion_status"] == "passed_encoder_gate"
    assert (output_dir / "per_spectrum_metrics.csv").exists()
    assert (output_dir / "aggregate_metrics.csv").exists()
    assert (output_dir / "paired_deltas.csv").exists()
    assert (output_dir / "stratified_metrics.csv").exists()
