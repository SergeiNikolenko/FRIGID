"""The join between the fitted corruption model and the trainer.

Three things are asserted here and nothing else is:

1. ``symmetric`` and ``dropout`` still draw exactly the numbers they drew
   before the fitted branch existed. The frozen-loss identity in
   ``test_marlin_training_recipe.py`` covers the objective; this file covers
   the corruption call itself, argument for argument.
2. ``fitted`` and ``fitted_control`` reach the model that
   ``scripts/fit_encoder_error_model.py`` wrote, and the control differs from
   the fitted law in exactly the frequency dependence and the row latent.
3. **Stage 2 inherits stage 1's corruption law.** Stage 2 is four times longer
   than stage 1, so a stage 2 that reverts to the incumbent noise spends 80% of
   its optimizer steps erasing stage 1 and the paired arms converge by
   construction. Inheriting is the default and changing it has to be an act.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from marlin.corpus_stream import (
    PACKAGED_HOLDOUT_INCHIKEYS,
    PACKAGED_HOLDOUT_INCHIKEYS_SHA256,
    resolve_repository_path,
)
from marlin.encoder_error_model import FINGERPRINT_BITS, EncoderErrorModel
from marlin.model import MarlinDecoderConfig
from marlin.noise import symmetric_fingerprint_noise
from marlin.training import (
    FINGERPRINT_NOISE_MODES,
    MarlinLightningModule,
    checkpoint_fingerprint_corruption,
    fingerprint_corruption,
    resolve_inherited_corruption,
    sha256_file,
)


def _config() -> MarlinDecoderConfig:
    return MarlinDecoderConfig(
        vocab_size=8,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=8,
        block_width=4,
        fingerprint_bits=FINGERPRINT_BITS,
    )


def _model(tmp_path, *, bins: int = 4) -> str:
    """A small fitted model on disk, with the frequency response the real one has."""
    bin_of = np.repeat(np.arange(bins), FINGERPRINT_BITS // bins)
    sensitivity = np.linspace(0.25, 0.99, bins)
    false_positive_rate = np.linspace(0.001, 0.85, bins)
    model = EncoderErrorModel(
        bin_of=bin_of.astype(np.int64),
        sensitivity=sensitivity,
        false_positive_rate=false_positive_rate,
        recall_latents=np.array([-1.0, 0.0, 1.0]),
        invention_latents=np.array([-0.5, 0.0, 0.5]),
        metadata={
            "name": "test",
            "bin_true_on_counts": [1000.0] * bins,
            "bin_true_off_counts": [1000.0] * bins,
        },
    )
    path = tmp_path / "encoder_error_model.npz"
    model.save(path)
    return str(path)


def _fingerprints(rows: int = 64) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    bits = torch.zeros((rows, FINGERPRINT_BITS))
    index = torch.randint(
        0, FINGERPRINT_BITS, (rows, 40), generator=generator
    )
    bits.scatter_(1, index, 1.0)
    return bits


# ----------------------------------------------------------------------
# 1. the incumbent laws are untouched
# ----------------------------------------------------------------------
def test_symmetric_still_draws_exactly_what_it_drew_before():
    module = MarlinLightningModule(_config(), fingerprint_noise_mode="symmetric")
    fingerprints = _fingerprints(8)

    torch.manual_seed(0)
    through_the_module = module.corrupt_conditioning(fingerprints)
    torch.manual_seed(0)
    directly = symmetric_fingerprint_noise(
        fingerprints,
        corruption_probability=module.noise_probability,
        min_fraction=module.noise_min_fraction,
        max_fraction=module.noise_max_fraction,
    )

    assert torch.equal(through_the_module, directly)


def test_the_incumbent_modes_hold_no_error_model():
    for mode in ("symmetric", "dropout"):
        module = MarlinLightningModule(_config(), fingerprint_noise_mode=mode)
        assert module.encoder_error_model is None


def test_an_error_model_on_an_incumbent_mode_is_refused(tmp_path):
    with pytest.raises(ValueError) as error:
        MarlinLightningModule(
            _config(),
            fingerprint_noise_mode="symmetric",
            fingerprint_error_model=_model(tmp_path),
        )
    assert "fingerprint_error_model" in str(error.value)


# ----------------------------------------------------------------------
# 2. the fitted branch reaches the fitted model
# ----------------------------------------------------------------------
def test_the_switch_accepts_the_two_fitted_modes():
    assert set(FINGERPRINT_NOISE_MODES) == {
        "symmetric",
        "dropout",
        "fitted",
        "fitted_control",
    }


def test_fitted_without_a_model_is_refused():
    with pytest.raises(ValueError) as error:
        MarlinLightningModule(_config(), fingerprint_noise_mode="fitted")
    assert "fingerprint_error_model" in str(error.value)


def test_an_unknown_mode_is_still_refused():
    with pytest.raises(ValueError):
        MarlinLightningModule(_config(), fingerprint_noise_mode="one-sided")


def test_fitted_corrupts_rare_bits_harder_than_common_ones(tmp_path):
    module = MarlinLightningModule(
        _config(),
        fingerprint_noise_mode="fitted",
        fingerprint_error_model=_model(tmp_path),
        noise_probability=1.0,
    )
    quarter = FINGERPRINT_BITS // 4
    fingerprints = torch.zeros((512, FINGERPRINT_BITS))
    fingerprints[:, :quarter] = 1.0  # the rarest bin
    fingerprints[:, -quarter:] = 1.0  # the commonest bin

    torch.manual_seed(3)
    corrupted = module.corrupt_conditioning(fingerprints)

    rare_recall = corrupted[:, :quarter].mean().item()
    common_recall = corrupted[:, -quarter:].mean().item()
    # The whole point of the fitted law: the informative rare bits are the ones
    # the encoder loses. The incumbent noise is flat at ~0.80 everywhere.
    assert common_recall > rare_recall + 0.3


def test_the_control_removes_the_frequency_dependence_and_holds_the_rate(tmp_path):
    path = _model(tmp_path)
    fitted = fingerprint_corruption("fitted", path)
    control = fingerprint_corruption("fitted_control", path)

    assert fitted.bins == 4
    assert control.bins == 1
    assert control.recall_latents.tolist() == [0.0]

    quarter = FINGERPRINT_BITS // 4
    fingerprints = torch.zeros((512, FINGERPRINT_BITS))
    fingerprints[:, :quarter] = 1.0
    fingerprints[:, -quarter:] = 1.0

    torch.manual_seed(5)
    under_fitted = fitted.corrupt(fingerprints, corruption_probability=1.0)
    torch.manual_seed(5)
    under_control = control.corrupt(fingerprints, corruption_probability=1.0)

    on = fingerprints > 0.5
    pooled_fitted = under_fitted[on].mean().item()
    pooled_control = under_control[on].mean().item()
    # Same pooled sensitivity, no frequency response: that difference, and only
    # that difference, is the ablation the pair is meant to price.
    assert abs(pooled_fitted - pooled_control) < 0.05
    rare = under_control[:, :quarter].mean().item()
    common = under_control[:, -quarter:].mean().item()
    assert abs(rare - common) < 0.05


def test_the_module_records_the_model_it_read(tmp_path):
    path = _model(tmp_path)
    module = MarlinLightningModule(
        _config(),
        fingerprint_noise_mode="fitted",
        fingerprint_error_model=path,
    )
    assert module.hparams["fingerprint_noise_mode"] == "fitted"
    assert module.hparams["fingerprint_error_model"] == path
    assert module.hparams["fingerprint_error_model_sha256"] == sha256_file(path)


def test_the_corruption_survives_a_checkpoint_round_trip(tmp_path):
    path = _model(tmp_path)
    module = MarlinLightningModule(
        _config(),
        fingerprint_noise_mode="fitted",
        fingerprint_error_model=path,
    )
    checkpoint = tmp_path / "stage1.ckpt"
    torch.save(
        {"state_dict": {}, "hyper_parameters": dict(module.hparams)}, checkpoint
    )

    recovered = checkpoint_fingerprint_corruption(checkpoint)

    assert recovered["fingerprint_noise_mode"] == "fitted"
    assert recovered["fingerprint_error_model"] == path
    assert recovered["fingerprint_error_model_sha256"] == sha256_file(path)


# ----------------------------------------------------------------------
# 3. stage 2 inherits stage 1
# ----------------------------------------------------------------------
def _stage_one(tmp_path, mode: str = "fitted") -> dict[str, object]:
    path = _model(tmp_path)
    return {
        "fingerprint_noise_mode": mode,
        "fingerprint_error_model": path,
        "fingerprint_error_model_sha256": sha256_file(path),
        "noise_probability": 1.0,
    }


def test_stage_two_inherits_stage_ones_corruption(tmp_path):
    stage_one = _stage_one(tmp_path)

    plan = resolve_inherited_corruption(
        requested_mode=None,
        requested_error_model=None,
        stage_one=stage_one,
    )

    assert plan["fingerprint_noise_mode"] == "fitted"
    assert plan["fingerprint_error_model"] == stage_one["fingerprint_error_model"]
    assert plan["changed_from_stage_one"] is False


def test_stage_two_may_not_silently_revert_to_the_incumbent_noise(tmp_path):
    """The reviewer's finding, as a test.

    Stage 1 is 5,000 steps and stage 2 is 20,000. A stage 2 that runs the
    incumbent symmetric law spends 80% of the pair's optimizer steps under a
    corruption whose KS distance to the real held-out DreaMS error is 0.8727
    against the fitted law's 0.0738, so the fitted arm and its control would
    differ over one fifth of their steps and converge.
    """
    with pytest.raises(ValueError) as error:
        resolve_inherited_corruption(
            requested_mode="symmetric",
            requested_error_model=None,
            stage_one=_stage_one(tmp_path),
        )
    message = str(error.value)
    assert "erases stage 1" in message
    assert "--allow-corruption-mode-change" in message


def test_changing_the_law_is_possible_when_it_is_the_experiment(tmp_path):
    plan = resolve_inherited_corruption(
        requested_mode="symmetric",
        requested_error_model=None,
        stage_one=_stage_one(tmp_path),
        allow_change=True,
    )
    assert plan["fingerprint_noise_mode"] == "symmetric"
    assert plan["changed_from_stage_one"] is True


def test_the_two_fitted_arms_do_not_inherit_each_others_law(tmp_path):
    """A ``fitted`` stage 1 followed by a ``fitted_control`` stage 2 is a mixed
    arm, and mixed arms are exactly what makes A and B converge."""
    with pytest.raises(ValueError):
        resolve_inherited_corruption(
            requested_mode="fitted_control",
            requested_error_model=_model(tmp_path),
            stage_one=_stage_one(tmp_path, mode="fitted"),
        )


def test_a_different_fitted_file_is_also_a_change(tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError):
        resolve_inherited_corruption(
            requested_mode="fitted",
            requested_error_model=_model(other, bins=8),
            stage_one=_stage_one(tmp_path, mode="fitted"),
        )


def test_a_checkpoint_that_records_nothing_falls_back_to_the_incumbent():
    plan = resolve_inherited_corruption(
        requested_mode=None,
        requested_error_model=None,
        stage_one=None,
    )
    assert plan["fingerprint_noise_mode"] == "symmetric"
    assert plan["fingerprint_error_model"] is None
    assert plan["changed_from_stage_one"] is False


def test_an_explicit_mode_matching_stage_one_is_not_a_change(tmp_path):
    stage_one = _stage_one(tmp_path, mode="symmetric")
    stage_one["fingerprint_error_model"] = None
    stage_one["fingerprint_error_model_sha256"] = None

    plan = resolve_inherited_corruption(
        requested_mode="symmetric",
        requested_error_model=None,
        stage_one=stage_one,
    )

    assert plan["changed_from_stage_one"] is False


# ----------------------------------------------------------------------
# the exclusion list travels with the code
# ----------------------------------------------------------------------
def test_the_holdout_list_is_inside_the_repository():
    assert PACKAGED_HOLDOUT_INCHIKEYS.exists()
    assert sha256_file(PACKAGED_HOLDOUT_INCHIKEYS) == PACKAGED_HOLDOUT_INCHIKEYS_SHA256
    blocks = PACKAGED_HOLDOUT_INCHIKEYS.read_text().splitlines()
    assert blocks[0] == "inchikey"
    assert len(blocks) - 1 == 1095


def test_a_repo_relative_path_does_not_depend_on_the_working_directory():
    resolved = resolve_repository_path("data/nplib1_holdout_inchikeys_v2.csv")
    assert resolved == PACKAGED_HOLDOUT_INCHIKEYS
    assert resolve_repository_path("/tmp/list.csv").as_posix() == "/tmp/list.csv"
