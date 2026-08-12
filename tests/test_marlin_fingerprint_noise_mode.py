"""One-sided training corruption, aligned with what inference does.

Inference draws its candidates by `perturb_fingerprint`, which only *removes*
on-bits. Training corrupted symmetrically, which keeps the cardinality and
invents bits the encoder never predicted, so the decoder was taught to trust a
conditioning vector of a shape it never receives.
"""

from __future__ import annotations

import torch

from marlin.noise import one_sided_fingerprint_dropout, symmetric_fingerprint_noise
from marlin.training import MarlinLightningModule
from marlin.model import MarlinDecoderConfig


def _fingerprints() -> torch.Tensor:
    fingerprints = torch.zeros((4, 64))
    fingerprints[:, :20] = 1.0
    return fingerprints


def test_dropout_only_removes_bits():
    fingerprints = _fingerprints()

    noisy = one_sided_fingerprint_dropout(
        fingerprints,
        corruption_probability=1.0,
        min_fraction=0.3,
        max_fraction=0.3,
        generator=torch.Generator().manual_seed(0),
    )

    assert torch.all(noisy <= fingerprints)
    assert not torch.any((noisy > 0.5) & (fingerprints < 0.5))
    assert noisy.sum() < fingerprints.sum()


def test_symmetric_noise_keeps_the_cardinality_dropout_does_not():
    fingerprints = _fingerprints()

    symmetric = symmetric_fingerprint_noise(
        fingerprints,
        corruption_probability=1.0,
        min_fraction=0.3,
        max_fraction=0.3,
        generator=torch.Generator().manual_seed(1),
    )
    dropped = one_sided_fingerprint_dropout(
        fingerprints,
        corruption_probability=1.0,
        min_fraction=0.3,
        max_fraction=0.3,
        generator=torch.Generator().manual_seed(1),
    )

    assert torch.equal(symmetric.sum(dim=1), fingerprints.sum(dim=1))
    assert torch.all(dropped.sum(dim=1) < fingerprints.sum(dim=1))


def test_zero_probability_leaves_the_fingerprint_alone():
    fingerprints = _fingerprints()

    noisy = one_sided_fingerprint_dropout(
        fingerprints,
        corruption_probability=0.0,
        generator=torch.Generator().manual_seed(2),
    )

    assert torch.equal(noisy, fingerprints)


def test_soft_amplitudes_survive_the_bits_that_are_kept():
    fingerprints = torch.zeros((1, 32))
    fingerprints[0, :8] = torch.linspace(0.6, 1.0, 8)

    noisy = one_sided_fingerprint_dropout(
        fingerprints,
        corruption_probability=1.0,
        min_fraction=0.5,
        max_fraction=0.5,
        generator=torch.Generator().manual_seed(3),
    )

    kept = noisy > 0.0
    assert torch.equal(noisy[kept], fingerprints[kept])


def test_module_refuses_an_unknown_noise_mode():
    config = MarlinDecoderConfig(
        vocab_size=8,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=4,
        block_width=2,
        fingerprint_bits=4,
        dropout=0.0,
        mask_token_id=3,
        pad_token_id=0,
    )
    try:
        MarlinLightningModule(config, fingerprint_noise_mode="one-sided")
    except ValueError as error:
        assert "fingerprint_noise_mode" in str(error)
    else:  # pragma: no cover - the guard is the point of the test
        raise AssertionError("an unknown corruption mode must be refused")
