"""CPU contract checks for the opt-in DreaMS fingerprint handoff."""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frigid.dreams_fingerprint_head import (  # noqa: E402
    FingerprintHead,
    checkpoint_payload,
    compute_fingerprint_metrics,
)
from frigid.fingerprint_backends import PrecomputedFingerprintBackend  # noqa: E402


def test_prediction_cli_checkpoint_and_named_spectrum_handoff(tmp_path):
    torch.manual_seed(7)
    model = FingerprintHead(4, fingerprint_bits=8, hidden_dim=16, dropout=0)
    model.eval()
    embeddings = torch.randn(3, 4)
    with torch.no_grad():
        expected = torch.sigmoid(model(embeddings)).numpy()
    checkpoint = tmp_path / "head.pt"
    torch.save(
        checkpoint_payload(
            model, input_dim=4, fingerprint_bits=8, hidden_dim=16,
            dropout=0, threshold=0.5, metrics={},
        ),
        checkpoint,
    )
    np.savez(tmp_path / "embeddings.npz", embeddings=embeddings.numpy())
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("spec_name\nspectrum-c\nspectrum-a\nspectrum-b\n")
    output = tmp_path / "predictions"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/predict_dreams_fingerprint_head.py"),
         "--checkpoint", str(checkpoint),
         "--embeddings-npz", str(tmp_path / "embeddings.npz"),
         "--metadata-csv", str(metadata), "--output-dir", str(output),
         "--batch-size", "2"],
        check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    backend = PrecomputedFingerprintBackend(
        str(output / "fingerprints.npz"), str(output / "metadata.csv")
    )
    np.testing.assert_allclose(backend.predict_probs({}, spec_name="spectrum-a"), expected[1])
    np.testing.assert_allclose(backend.predict_probs({}, spec_name="spectrum-c"), expected[0])
    with pytest.raises(KeyError, match="not found"):
        backend.predict_probs({}, spec_name="missing")


def test_precomputed_handoff_rejects_misaligned_rows(tmp_path):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("spec_name\na\nb\n")
    np.savez(tmp_path / "fingerprints.npz", probs=np.ones((1, 8)))
    with pytest.raises(ValueError, match="do not match"):
        PrecomputedFingerprintBackend(str(tmp_path / "fingerprints.npz"), str(metadata))


def test_fingerprint_metrics_keep_exact_and_disjoint_cases_distinct():
    metrics, per_case = compute_fingerprint_metrics(
        np.array([[0.9, 0.1], [0.1, 0.9]]),
        np.array([[1, 0], [1, 0]]), 0.5,
    )
    np.testing.assert_array_equal(per_case, [1, 0])
    assert metrics.mean_tanimoto == 0.5
    assert metrics.bit_precision == metrics.bit_recall == 0.5
