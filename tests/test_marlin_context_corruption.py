"""The corrupted-context objective, and the identity that makes its control free.

At sampling time the clean stream carries the *committed* prefix, so one wrong
committed token becomes the context of every later block; in training the clean
stream is always gold. `context_corruption_probability` closes that gap. The
first test below is the one that matters operationally: at probability 0 the
objective must be bit-identical to the historical one, including the generator
state it consumes, or the paired control arm is not a control.
"""

from __future__ import annotations

import torch

from marlin.model import MarlinDecoder, MarlinDecoderConfig


def _config() -> MarlinDecoderConfig:
    return MarlinDecoderConfig(
        vocab_size=8,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=6,
        block_width=2,
        fingerprint_bits=4,
        dropout=0.0,
        mask_token_id=3,
        pad_token_id=0,
    )


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[1, 4, 5, 6, 7, 2], [1, 5, 4, 6, 2, 0]]),
        torch.tensor([50.0, 60.0]),
        torch.zeros((2, 4)),
    )


def test_zero_probability_reproduces_the_historical_objective():
    torch.manual_seed(0)
    model = MarlinDecoder(_config())
    clean_ids, mass, fingerprint = _batch()

    baseline_generator = torch.Generator().manual_seed(11)
    baseline_loss, baseline_metrics = model.diffusion_objective(
        clean_ids,
        mass,
        fingerprint,
        generator=baseline_generator,
        collect_metrics=True,
    )
    corrupted_generator = torch.Generator().manual_seed(11)
    corrupted_loss, corrupted_metrics = model.diffusion_objective(
        clean_ids,
        mass,
        fingerprint,
        generator=corrupted_generator,
        context_corruption_probability=0.0,
        context_corruption_min_fraction=0.05,
        context_corruption_max_fraction=0.25,
        restoration_loss_weight=0.5,
        confusion_ids=torch.full_like(clean_ids, 4),
        collect_metrics=True,
    )

    assert torch.equal(baseline_loss, corrupted_loss)
    # The corruption must not have drawn from the generator either, or a p=0 run
    # would diverge from the control after the first step.
    assert torch.equal(baseline_generator.get_state(), corrupted_generator.get_state())
    assert float(corrupted_metrics["context_corruption_fraction"]) == 0.0
    assert float(corrupted_metrics["restoration_token_count"]) == 0.0
    for key, value in baseline_metrics.items():
        assert torch.equal(value, corrupted_metrics[key]), key


def test_corruption_requires_confusion_ids():
    model = MarlinDecoder(_config())
    clean_ids, mass, fingerprint = _batch()
    try:
        model.diffusion_objective(
            clean_ids,
            mass,
            fingerprint,
            generator=torch.Generator().manual_seed(3),
            context_corruption_probability=0.5,
        )
    except ValueError as error:
        assert "confusion_ids" in str(error)
    else:  # pragma: no cover - the guard is the point of the test
        raise AssertionError("a corrupted context without confusions must be refused")


def test_corruption_scores_visible_wrong_tokens_against_gold():
    torch.manual_seed(0)
    model = MarlinDecoder(_config())
    clean_ids, mass, fingerprint = _batch()
    confusion_ids = torch.where(
        clean_ids.eq(4), torch.full_like(clean_ids, 5), torch.full_like(clean_ids, 4)
    )

    loss, metrics = model.diffusion_objective(
        clean_ids,
        mass,
        fingerprint,
        generator=torch.Generator().manual_seed(5),
        context_corruption_probability=1.0,
        context_corruption_min_fraction=0.9,
        context_corruption_max_fraction=1.0,
        restoration_loss_weight=0.5,
        confusion_ids=confusion_ids,
        collect_metrics=True,
    )

    assert torch.isfinite(loss)
    assert float(metrics["context_corruption_fraction"]) > 0.0
    assert float(metrics["restoration_token_count"]) > 0.0
    assert 0.0 <= float(metrics["restoration_token_accuracy_top1"]) <= 1.0
    assert 0.0 <= float(metrics["restoration_copy_rate"]) <= 1.0


def test_confusions_are_never_the_gold_token_where_the_model_is_right():
    torch.manual_seed(0)
    model = MarlinDecoder(_config())
    clean_ids, mass, fingerprint = _batch()

    confusions = model.sample_confusions(clean_ids, mass, fingerprint)

    assert confusions.shape == clean_ids.shape
    content = clean_ids.ne(model.config.pad_token_id)
    content[:, 0] = False
    assert not torch.any(confusions[content].eq(clean_ids[content]) & content[content])
