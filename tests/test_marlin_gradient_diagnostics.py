import math

import pytest
import torch

from marlin.gradient_diagnostics import (
    PARAMETER_GROUPS,
    EmaDivergence,
    GradientDiagnostics,
    ema_divergence,
    gradient_group_norms,
    parameter_group,
)
from marlin.model import MarlinDecoderConfig
from marlin.training import MarlinLightningModule


def _tiny_module():
    config = MarlinDecoderConfig(
        vocab_size=16,
        hidden_size=8,
        num_layers=2,
        num_heads=1,
        intermediate_size=16,
        max_length=8,
        block_width=2,
        fingerprint_bits=8,
        dropout=0.0,
        mask_token_id=3,
        pad_token_id=0,
    )
    return MarlinLightningModule(config, ema_decay=0.5)


def _backward(module):
    fingerprint = torch.zeros((2, 8))
    fingerprint[0, [1, 4]] = 1.0
    fingerprint[1, [2, 6]] = 1.0
    loss, _ = module.decoder.diffusion_objective(
        torch.tensor([[1, 5, 6, 7, 2], [1, 7, 8, 9, 2]]),
        torch.tensor([120.0, 140.0]),
        fingerprint,
        isotope_ratios=torch.zeros((2, 2)),
        full_sequence_mask_probability=1.0,
    )
    loss.backward()


def test_parameter_group_splits_the_conditioning_pathway_from_the_backbone():
    assert parameter_group("decoder.conditioner.fingerprint.embedding.weight") == (
        "fingerprint_conditioner"
    )
    assert parameter_group("conditioner.mass.projection.0.weight") == "mass_conditioner"
    assert parameter_group("conditioner.isotope.0.bias") == "isotope_conditioner"
    # norm2 belongs to the cross-attention residual and is unfrozen with it.
    assert parameter_group("layers.3.cross_attention.in_proj_weight") == "cross_attention"
    assert parameter_group("layers.3.norm2.weight") == "cross_attention"
    assert parameter_group("layers.3.self_attention.out_proj.weight") == "backbone"
    assert parameter_group("layers.3.linear1.weight") == "backbone"
    assert parameter_group("token_embedding.weight") == "embedding"
    assert parameter_group("output_bias") == "head"


def test_every_decoder_parameter_lands_in_a_known_group():
    module = _tiny_module()

    groups = {parameter_group(name) for name, _ in module.named_parameters()}

    assert groups <= set(PARAMETER_GROUPS)
    # The three the architecture question is about must all be populated.
    assert {"fingerprint_conditioner", "cross_attention", "backbone"} <= groups


def test_group_norms_recompose_into_the_global_norm():
    """A share that does not sum to one would make "the conditioner is starved"
    unfalsifiable."""
    module = _tiny_module()
    _backward(module)

    values = gradient_group_norms(module.named_parameters())

    squares = sum(
        values[f"grad_norm/{group}"] ** 2
        for group in PARAMETER_GROUPS
        if f"grad_norm/{group}" in values
    )
    assert squares == pytest.approx(values["grad_norm/global"] ** 2, rel=1e-6)
    shares = sum(
        values[f"grad_share/{group}"]
        for group in PARAMETER_GROUPS
        if f"grad_share/{group}" in values
    )
    assert shares == pytest.approx(1.0, rel=1e-6)


def test_a_frozen_pathway_reads_zero_coverage():
    """Distinguish "this pathway received nothing" from "it received zero"."""
    module = _tiny_module()
    for name, parameter in module.named_parameters():
        if parameter_group(name) == "fingerprint_conditioner":
            parameter.requires_grad_(False)
    _backward(module)

    values = gradient_group_norms(module.named_parameters())

    assert values["grad_covered/fingerprint_conditioner"] == 0.0
    assert values["grad_norm/fingerprint_conditioner"] == 0.0
    assert values["grad_covered/backbone"] == 1.0
    assert values["grad_norm/backbone"] > 0.0


def test_no_gradients_at_all_report_a_finite_zero():
    module = _tiny_module()

    values = gradient_group_norms(module.named_parameters())

    assert values["grad_norm/global"] == 0.0
    assert values["grad_share/backbone"] == 0.0
    assert values["grad_parameters_with_gradient"] == 0.0


class _Trainer:
    def __init__(self):
        self.global_step = 0
        self.is_global_zero = True


class _Module(torch.nn.Module):
    def __init__(self, norm: float):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4))
        self.weight.grad = torch.full((4,), norm / 2.0)
        self.logged: dict[str, float] = {}

    def log(self, name, value, **kwargs):
        self.logged[name] = float(value)


class _Logger:
    def __init__(self):
        self.scalars = []

    def report_scalar(self, **kwargs):
        self.scalars.append(kwargs)


def _task(logger):
    return type("Task", (), {"get_logger": lambda self: logger})()


def test_clip_rate_counts_every_optimizer_step_not_only_the_reported_ones():
    """A clipping rate sampled every fiftieth step is a subsample, not a rate."""
    callback = GradientDiagnostics(interval_steps=4, clip_value=1.0)
    trainer = _Trainer()
    module = _Module(norm=4.0)

    for step in range(1, 5):
        trainer.global_step = step
        callback.on_before_optimizer_step(trainer, module, None)

    assert callback.optimizer_steps == 4
    assert callback.clipped_steps == 4
    assert module.logged["grad_clip_rate"] == pytest.approx(1.0)
    assert module.logged["grad_clip_scale"] == pytest.approx(0.25)
    assert module.logged["grad_clipped"] == 1.0


def test_clip_rate_mixes_clipped_and_unclipped_steps():
    callback = GradientDiagnostics(interval_steps=1, clip_value=1.0)
    trainer = _Trainer()
    large = _Module(norm=4.0)
    small = _Module(norm=0.4)

    for step, module in enumerate([large, small, small, small], start=1):
        trainer.global_step = step
        callback.on_before_optimizer_step(trainer, module, None)

    assert callback.clipped_steps == 1
    assert small.logged["grad_clip_rate"] == pytest.approx(0.25)
    assert small.logged["grad_clip_scale"] == pytest.approx(1.0)


def test_gradient_diagnostics_reports_only_on_its_interval():
    logger = _Logger()
    callback = GradientDiagnostics(
        interval_steps=10, clip_value=1.0, clearml_task=_task(logger)
    )
    trainer = _Trainer()
    module = _Module(norm=0.4)

    for step in (1, 5, 10, 10, 20):
        trainer.global_step = step
        callback.on_before_optimizer_step(trainer, module, None)

    # Every step is counted, only two are published.
    assert callback.optimizer_steps == 5
    assert sorted({entry["iteration"] for entry in logger.scalars}) == [10, 20]
    assert "grad_norm/global" in {entry["series"] for entry in logger.scalars}


def test_gradient_diagnostics_refuses_an_impossible_configuration():
    with pytest.raises(ValueError, match="interval must be positive"):
        GradientDiagnostics(interval_steps=0)
    with pytest.raises(ValueError, match="clip_value must be positive"):
        GradientDiagnostics(interval_steps=1, clip_value=0.0)


def test_ema_divergence_is_zero_at_the_reset_and_grows_with_the_weights():
    module = _tiny_module()

    start = ema_divergence(module.ema.shadow_params, module.decoder.parameters())
    assert start["ema_distance"] == pytest.approx(0.0)
    assert start["ema_relative_distance"] == pytest.approx(0.0)

    with torch.no_grad():
        for parameter in module.decoder.parameters():
            parameter.add_(0.1)
    moved = ema_divergence(module.ema.shadow_params, module.decoder.parameters())

    count = sum(p.numel() for p in module.decoder.parameters())
    assert moved["ema_distance"] == pytest.approx(0.1 * math.sqrt(count), rel=1e-5)
    assert moved["ema_relative_distance"] > 0.0


def test_ema_divergence_refuses_a_mismatched_shadow():
    module = _tiny_module()

    with pytest.raises(ValueError, match="does not match the parameters"):
        ema_divergence(module.ema.shadow_params[:-1], module.decoder.parameters())


def test_ema_divergence_callback_reports_on_its_interval():
    logger = _Logger()
    callback = EmaDivergence(interval_steps=10, clearml_task=_task(logger))
    module = _tiny_module()
    module.log = lambda name, value, **kwargs: None
    trainer = _Trainer()

    for step in (0, 5, 10, 10, 20):
        trainer.global_step = step
        callback.on_train_batch_end(trainer, module, None, None, 0)

    assert sorted({entry["iteration"] for entry in logger.scalars}) == [10, 20]
    assert {entry["series"] for entry in logger.scalars} == {
        "ema_distance",
        "ema_parameter_norm",
        "ema_relative_distance",
    }


class _Loader(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        fingerprint = torch.zeros(8)
        fingerprint[[index % 8, (index + 3) % 8]] = 1.0
        return {
            "input_ids": torch.tensor([1, 5, 6, 7, 2]),
            "fingerprint": fingerprint,
            "precursor_mass": torch.tensor(120.0 + index),
            "isotope_ratios": torch.zeros(2),
        }


def test_the_callbacks_fire_inside_a_real_trainer_before_clipping():
    """Lightning calls ``on_before_optimizer_step`` immediately before
    ``_clip_gradients``, so the norms reported here are the ones the clip
    threshold is compared against."""
    import lightning as L

    module = _tiny_module()
    logger = _Logger()
    gradients = GradientDiagnostics(
        interval_steps=1, clip_value=1e-9, clearml_task=_task(logger)
    )
    ema = EmaDivergence(interval_steps=1, clearml_task=_task(logger))
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=2,
        gradient_clip_val=1e-9,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[gradients, ema],
    )

    trainer.fit(
        module,
        torch.utils.data.DataLoader(_Loader(), batch_size=2),
    )

    assert gradients.optimizer_steps == 2
    # Everything is clipped at a 1e-9 threshold, and the norm seen by the
    # callback is the pre-clip one, so the rate is 1.
    assert gradients.clipped_steps == 2
    reported = {entry["series"] for entry in logger.scalars}
    assert "grad_norm/fingerprint_conditioner" in reported
    assert "grad_share/cross_attention" in reported
    assert "grad_clip_rate" in reported
    assert "ema_relative_distance" in reported
