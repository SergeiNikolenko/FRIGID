"""Tests for the fitted encoder error model and its rate-matched control."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from marlin.encoder_error_model import (
    FINGERPRINT_BITS,
    EncoderErrorModel,
    fit_encoder_error_model,
)


def _logit(p):
    return np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))


def _synthetic(rows: int = 400, seed: int = 0, row_sd: float = 1.2):
    """Errors that are frequency-dependent *and* vary by row, like the real ones.

    Both properties are in the real measurement: sensitivity runs 0.255 to 0.991
    across corpus-frequency deciles, and the per-row Tanimoto standard deviation
    is 0.151 against the 0.066 a row-homogeneous model produces. A fixture
    without the second property cannot test that the latent recovers it.
    """
    rng = np.random.default_rng(seed)
    frequency = np.exp(rng.normal(-6.0, 1.5, FINGERPRINT_BITS))
    frequency = np.clip(frequency, 1e-4, 0.95)
    truth = rng.random((rows, FINGERPRINT_BITS)) < (frequency * 1.2)[None, :]
    # sensitivity climbs with frequency, invention climbs faster
    sensitivity = np.clip(0.2 + 0.8 * (frequency / frequency.max()) ** 0.3, 0.05, 0.99)
    invention = np.clip(0.0005 + 0.9 * (frequency / frequency.max()) ** 1.5, 0.0, 0.9)
    recall_shift = rng.normal(0.0, row_sd, rows)[:, None]
    invent_shift = rng.normal(0.0, 0.5 * row_sd, rows)[:, None]
    on = 1 / (1 + np.exp(-(_logit(sensitivity)[None, :] + recall_shift)))
    off = 1 / (1 + np.exp(-(_logit(invention)[None, :] + invent_shift)))
    predicted = rng.random(truth.shape) < np.where(truth, on, off)
    return truth, predicted, frequency


def _tanimoto(left, right):
    intersection = (left & right).sum(1)
    union = (left | right).sum(1)
    return np.where(union > 0, intersection / np.maximum(union, 1), 0.0)


def test_fit_recovers_a_monotone_frequency_response():
    truth, predicted, frequency = _synthetic()
    model = fit_encoder_error_model(truth, predicted, frequency, bins=8)
    assert model.bins == 8
    # the fit must see the structure the incumbent flat noise cannot express
    assert model.sensitivity[-1] > model.sensitivity[0] + 0.2
    assert model.false_positive_rate[-1] > model.false_positive_rate[0] * 10


def test_bins_carry_comparable_true_on_mass():
    truth, predicted, frequency = _synthetic()
    model = fit_encoder_error_model(truth, predicted, frequency, bins=10)
    mass = np.asarray(model.metadata["bin_true_on_counts"])
    assert mass.min() > 0
    # equal-mass binning, not equal-width: no bin may dominate
    assert mass.max() / mass.min() < 3.0


def test_sampled_marginals_match_the_fit():
    truth, predicted, frequency = _synthetic(rows=600, seed=3)
    model = fit_encoder_error_model(truth, predicted, frequency, bins=10)
    rng = torch.Generator().manual_seed(0)
    tiled = np.tile(truth, (5, 1))
    sampled = model.corrupt(
        torch.from_numpy(tiled.astype(np.float32)), generator=rng
    ).numpy() > 0.5
    real_tp = (truth & predicted).sum(1).mean()
    real_fp = ((~truth) & predicted).sum(1).mean()
    assert (tiled & sampled).sum(1).mean() == pytest.approx(real_tp, rel=0.06)
    assert ((~tiled) & sampled).sum(1).mean() == pytest.approx(real_fp, rel=0.10)


def test_row_latent_reproduces_the_dispersion_a_flat_model_loses():
    truth, predicted, frequency = _synthetic(rows=600, seed=5)
    model = fit_encoder_error_model(truth, predicted, frequency, bins=10)
    rng = torch.Generator().manual_seed(1)
    tiled = np.tile(truth, (5, 1))
    tensor = torch.from_numpy(tiled.astype(np.float32))
    fitted_sd = _tanimoto(tiled, model.corrupt(tensor, generator=rng).numpy() > 0.5).std()
    flat = model.rate_matched_uniform_control()
    flat_sd = _tanimoto(tiled, flat.corrupt(tensor, generator=rng).numpy() > 0.5).std()
    real_sd = _tanimoto(truth, predicted).std()
    assert abs(fitted_sd - real_sd) < abs(flat_sd - real_sd)


def test_control_matches_the_pooled_rates_but_not_the_structure():
    truth, predicted, frequency = _synthetic()
    model = fit_encoder_error_model(truth, predicted, frequency, bins=10)
    control = model.rate_matched_uniform_control()
    assert control.bins == 1
    assert np.all(control.bin_of == 0)
    rng = torch.Generator().manual_seed(2)
    tiled = np.tile(truth, (4, 1))
    tensor = torch.from_numpy(tiled.astype(np.float32))
    fitted = model.corrupt(tensor, generator=rng).numpy() > 0.5
    flat = control.corrupt(
        tensor, generator=torch.Generator().manual_seed(21)
    ).numpy() > 0.5
    # same amount of error ...
    assert (tiled & flat).sum(1).mean() == pytest.approx(
        (tiled & fitted).sum(1).mean(), rel=0.08
    )
    assert ((~tiled) & flat).sum(1).mean() == pytest.approx(
        ((~tiled) & fitted).sum(1).mean(), rel=0.12
    )
    # ... spent on different bits: the control cannot prefer the frequent ones
    common = np.argsort(frequency)[-64:]
    assert flat[:, common].mean() < fitted[:, common].mean()


def test_corruption_probability_zero_is_the_identity():
    truth, predicted, frequency = _synthetic(rows=64)
    model = fit_encoder_error_model(truth, predicted, frequency, bins=4)
    tensor = torch.from_numpy(truth.astype(np.float32))
    out = model.corrupt(
        tensor,
        corruption_probability=0.0,
        generator=torch.Generator().manual_seed(0),
    )
    assert torch.equal(out, tensor)


def test_corruption_is_seed_reproducible():
    truth, predicted, frequency = _synthetic(rows=64)
    model = fit_encoder_error_model(truth, predicted, frequency, bins=4)
    tensor = torch.from_numpy(truth.astype(np.float32))
    first = model.corrupt(tensor, generator=torch.Generator().manual_seed(7))
    second = model.corrupt(tensor, generator=torch.Generator().manual_seed(7))
    assert torch.equal(first, second)


def test_output_is_binary_and_shaped_like_the_input():
    truth, predicted, frequency = _synthetic(rows=32)
    model = fit_encoder_error_model(truth, predicted, frequency, bins=4)
    tensor = torch.from_numpy(truth.astype(np.float32))
    out = model.corrupt(tensor, generator=torch.Generator().manual_seed(0))
    assert out.shape == tensor.shape
    assert out.dtype == tensor.dtype
    assert set(torch.unique(out).tolist()) <= {0.0, 1.0}


def test_round_trip_through_disk(tmp_path):
    truth, predicted, frequency = _synthetic(rows=64)
    model = fit_encoder_error_model(truth, predicted, frequency, bins=6)
    path = tmp_path / "model.npz"
    model.save(path)
    restored = EncoderErrorModel.load(path)
    assert np.array_equal(restored.bin_of, model.bin_of)
    assert np.allclose(restored.sensitivity, model.sensitivity)
    assert np.allclose(restored.false_positive_rate, model.false_positive_rate)
    assert np.allclose(restored.recall_latents, model.recall_latents)
    assert restored.metadata["bins"] == model.metadata["bins"]
    tensor = torch.from_numpy(truth.astype(np.float32))
    a = model.corrupt(tensor, generator=torch.Generator().manual_seed(3))
    b = restored.corrupt(tensor, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_rejects_a_fingerprint_of_the_wrong_width():
    truth, predicted, frequency = _synthetic(rows=8)
    model = fit_encoder_error_model(truth, predicted, frequency, bins=4)
    with pytest.raises(ValueError, match="fitted for 4096 bits"):
        model.corrupt(torch.zeros(2, 2048))
    with pytest.raises(ValueError, match=r"\[batch, bits\]"):
        model.corrupt(torch.zeros(4096))


def test_rejects_a_corpus_frequency_of_the_wrong_width():
    truth, predicted, _ = _synthetic(rows=8)
    with pytest.raises(ValueError, match="corpus_bit_frequency"):
        fit_encoder_error_model(truth, predicted, np.zeros(128))


def test_rejects_an_impossible_corruption_probability():
    truth, predicted, frequency = _synthetic(rows=8)
    model = fit_encoder_error_model(truth, predicted, frequency, bins=2)
    with pytest.raises(ValueError, match="corruption_probability"):
        model.corrupt(torch.zeros(2, FINGERPRINT_BITS), corruption_probability=1.5)
