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

    arguments = [
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
            "--external-training-overlap",
            "candidate=checked",
            "--stratify-column",
            "adduct",
            "--code-revision",
            "test-revision",
            "--output-dir",
            str(output_dir),
        ]
    exit_code = MODULE.main(arguments)

    assert exit_code == 0
    summary = json.loads((output_dir / "benchmark_summary.json").read_text())
    assert summary["baseline"] == "mist"
    assert summary["benchmark_code_revision"] == "test-revision"
    assert [row["model"] for row in summary["ranking"]] == ["candidate", "mist"]
    assert summary["ranking"][0]["promotion_status"] == "passed_encoder_gate"
    assert summary["ranking"][0]["external_training_overlap_status"] == "checked"
    assert (output_dir / "per_spectrum_metrics.csv").exists()
    assert (output_dir / "aggregate_metrics.csv").exists()
    assert (output_dir / "paired_deltas.csv").exists()
    assert (output_dir / "stratified_metrics.csv").exists()

    undeclared_output = tmp_path / "undeclared_results"
    undeclared_arguments = arguments.copy()
    overlap_flag = undeclared_arguments.index("--external-training-overlap")
    del undeclared_arguments[overlap_flag : overlap_flag + 2]
    undeclared_arguments[-1] = str(undeclared_output)
    assert MODULE.main(undeclared_arguments) == 0
    undeclared_summary = json.loads(
        (undeclared_output / "benchmark_summary.json").read_text()
    )
    assert (
        undeclared_summary["ranking"][0]["promotion_status"]
        == "needs_external_pretraining_overlap_evidence"
    )

    overlapping_output = tmp_path / "overlapping_results"
    training_path.write_text("AAAAAAAAAAAAAA\n")
    overlapping_arguments = arguments.copy()
    overlapping_arguments[-1] = str(overlapping_output)
    assert MODULE.main(overlapping_arguments) == 0
    overlapping_summary = json.loads(
        (overlapping_output / "benchmark_summary.json").read_text()
    )
    assert (
        overlapping_summary["ranking"][0]["promotion_status"]
        == "failed_training_overlap_check"
    )


def test_cli_calibrates_thresholds_on_disjoint_partition(tmp_path: Path):
    metadata_path = tmp_path / "metadata.csv"
    fingerprints_path = tmp_path / "fingerprints.npz"
    candidate_path = tmp_path / "candidate.npz"
    selection_path = tmp_path / "partitions.csv"
    training_path = tmp_path / "candidate_train.txt"
    output_dir = tmp_path / "calibrated_results"

    pd.DataFrame(
        {
            "fingerprint_index": [0, 1],
            "spec_name": ["spec-a", "spec-b"],
            "inchi_key_first_block": ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"],
        }
    ).to_csv(metadata_path, index=False)
    np.savez_compressed(
        fingerprints_path,
        ground_truth=np.asarray([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=np.float32),
        mist_probs=np.asarray(
            [[0.9, 0.6, 0.8, 0.2], [0.6, 0.7, 0.1, 0.2]], dtype=np.float32
        ),
    )
    np.savez_compressed(
        candidate_path,
        spectrum_ids=np.asarray(["spec-b", "spec-a"]),
        probs=np.asarray(
            [[0.1, 0.9, 0.1, 0.1], [0.9, 0.1, 0.8, 0.2]], dtype=np.float32
        ),
    )
    pd.DataFrame(
        {
            "spec_name": ["spec-a", "spec-b"],
            "benchmark_partition": ["calibration", "evaluation"],
        }
    ).to_csv(selection_path, index=False)
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
            "--selection-manifest",
            str(selection_path),
            "--calibrate-thresholds",
            "--threshold-grid",
            "0.5",
            "--baseline",
            "mist",
            "--expected-bits",
            "4",
            "--bootstrap-samples",
            "20",
            "--training-identifiers",
            f"candidate={training_path}",
            "--external-training-overlap",
            "candidate=unknown",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "benchmark_summary.json").read_text())
    assert summary["selection"]["calibration_rows"] == 1
    assert summary["selection"]["evaluation_rows"] == 1
    assert summary["ranking"][0]["model"] == "candidate"
    assert summary["ranking"][0]["threshold_source"] == "calibration_partition"
    assert (
        summary["ranking"][0]["promotion_status"]
        == "needs_external_pretraining_overlap_evidence"
    )
    assert (output_dir / "threshold_calibration.csv").exists()
