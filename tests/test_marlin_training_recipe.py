"""The four recipe corrections, each in isolation, and the composite identity.

Every correction is opt-in, so the load-bearing test in this file is
``test_every_new_flag_off_is_the_recorded_training_step``: with the defaults the
loss is the frozen number the recipe produced before any of this existed, which
is what keeps the runs recorded under the old recipe comparable to the new ones.
"""

from __future__ import annotations

import math

import pytest
import torch

from marlin.checkpoint_selection import (
    PANEL_SELECTION_METRICS,
    PROBE_SELECTION_METRICS,
    CheckpointSelector,
)
from marlin.lr_schedule import (
    ADAMW_SECOND_MOMENT_WINDOW,
    RELEASED_COSINE_LAST_STEP,
    RELEASED_TERMINAL_LEARNING_RATE,
    RELEASED_WARMUP_STEPS,
    build_warmup_cosine_scheduler,
    derive_peak_learning_rate,
    released_final_decade_displacement,
    released_final_decade_steps,
    released_learning_rate,
    schedule_displacement,
    warmup_cosine_factor,
)
from marlin.model import (
    MarlinDecoder,
    MarlinDecoderConfig,
    antithetic_uniform_times,
)
from marlin.training import MarlinLightningModule, fp32_forward_context


# The step the whole file is anchored on: two rows of 17 tokens, 15 valid
# targets each, a two-layer decoder seeded at 11, and the objective's generator
# seeded at 1234.
FROZEN_BLOCK_MEAN_LOSS = 45.71464157104492
FROZEN_TOKEN_MEAN_LOSS = 6.095285892486572
FROZEN_ANTITHETIC_BLOCK_MEAN_LOSS = 29.83205795288086


def _fixture() -> tuple[MarlinDecoder, dict[str, torch.Tensor]]:
    torch.manual_seed(11)
    config = MarlinDecoderConfig(
        vocab_size=16,
        hidden_size=16,
        num_layers=2,
        num_heads=2,
        intermediate_size=32,
        max_length=17,
        block_width=8,
        fingerprint_bits=8,
        dropout=0.0,
        mask_token_id=4,
        pad_token_id=0,
        eos_token_id=2,
    )
    decoder = MarlinDecoder(config).eval()
    batch = {
        "input_ids": torch.tensor(
            [
                [1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 6, 7, 8, 9, 5, 2, 0],
                [1, 7, 8, 5, 6, 9, 11, 10, 12, 6, 7, 13, 5, 8, 9, 2, 0],
            ]
        ),
        "precursor_mass": torch.tensor([301.5, 244.25]),
        "fingerprint": torch.tensor(
            [[1.0, 0, 1, 0, 1, 1, 0, 0], [0, 1, 1, 1, 0, 0, 1, 0]]
        ),
        "isotope_ratios": torch.tensor([[0.11, 0.02], [0.09, 0.015]]),
    }
    return decoder, batch


def _loss(decoder, batch, **kwargs) -> float:
    generator = torch.Generator().manual_seed(1234)
    loss, _ = decoder.diffusion_objective(
        batch["input_ids"],
        batch["precursor_mass"],
        batch["fingerprint"],
        isotope_ratios=batch["isotope_ratios"],
        generator=generator,
        **kwargs,
    )
    return float(loss)


# --------------------------------------------------------------------------
# The composite: defaults are the recipe that has already run.
# --------------------------------------------------------------------------


def test_every_new_flag_off_is_the_recorded_training_step():
    decoder, batch = _fixture()
    assert _loss(decoder, batch) == pytest.approx(FROZEN_BLOCK_MEAN_LOSS, rel=1e-6)
    explicit = _loss(
        decoder,
        batch,
        time_sampling="per_block_iid",
        loss_reduction="block_mean",
    )
    assert explicit == pytest.approx(FROZEN_BLOCK_MEAN_LOSS, rel=1e-6)


def test_the_module_defaults_are_the_old_recipe():
    module = MarlinLightningModule(_fixture()[0].config)
    assert module.lr_schedule == "constant"
    assert module.time_sampling == "per_block_iid"
    assert module.loss_reduction == "block_mean"
    assert module.fp32_forward is False
    # A constant schedule returns a bare optimizer, exactly as before, so
    # Lightning wires no scheduler at all.
    assert isinstance(module.configure_optimizers(), torch.optim.AdamW)


def test_turning_every_flag_on_changes_the_step():
    decoder, batch = _fixture()
    corrected = _loss(
        decoder,
        batch,
        time_sampling="per_sequence_antithetic",
        loss_reduction="token_mean",
    )
    assert corrected != pytest.approx(FROZEN_BLOCK_MEAN_LOSS, rel=1e-3)


# --------------------------------------------------------------------------
# (b) loss reduction
# --------------------------------------------------------------------------


def test_token_mean_is_the_block_mean_times_tokens_per_block():
    """15 valid targets per row against ceil(15/8) = 2 blocks is a 7.5x factor.

    The rows are equal length, so the two reductions differ by exactly that
    ratio and the ``~8x`` in docs/TRAINING_RECIPE_FINDINGS.md:20 is exact here.
    """
    decoder, batch = _fixture()
    valid = batch["input_ids"].ne(decoder.config.pad_token_id)
    valid[:, 0] = False
    per_row_valid = int(valid.sum(dim=1)[0])
    blocks = math.ceil(per_row_valid / decoder.config.block_width)
    block_mean = _loss(decoder, batch, loss_reduction="block_mean")
    token_mean = _loss(decoder, batch, loss_reduction="token_mean")
    assert block_mean == pytest.approx(FROZEN_BLOCK_MEAN_LOSS, rel=1e-6)
    assert token_mean == pytest.approx(FROZEN_TOKEN_MEAN_LOSS, rel=1e-6)
    assert token_mean * per_row_valid == pytest.approx(block_mean * blocks, rel=1e-5)
    assert block_mean / token_mean == pytest.approx(per_row_valid / blocks, rel=1e-5)


def test_token_mean_weights_a_long_row_by_its_length():
    """The checkpoint's global_mean_loss is one sum over the batch.

    A padded short row must therefore contribute fewer tokens, not an equal
    share. The block mean cannot tell the two apart once both rows round up to
    the same block count, which is the defect.
    """
    decoder, batch = _fixture()
    shortened = {name: value.clone() for name, value in batch.items()}
    shortened["input_ids"][1, 10:] = decoder.config.pad_token_id
    assert _loss(decoder, shortened, loss_reduction="token_mean") != pytest.approx(
        _loss(decoder, batch, loss_reduction="token_mean"), rel=1e-6
    )


def test_an_unknown_reduction_is_refused():
    decoder, batch = _fixture()
    with pytest.raises(ValueError, match="unknown loss_reduction"):
        _loss(decoder, batch, loss_reduction="mean")


# --------------------------------------------------------------------------
# (c) time sampling
# --------------------------------------------------------------------------


def _released_time_distribution(count: int, *, seed: int, sampling_eps: float):
    """An independent transcription of src/dlm/utils/utils_moco.py:82-86."""
    generator = torch.Generator().manual_seed(seed)
    time_step = torch.rand(count, generator=generator)
    offset = torch.arange(count) / count
    time_step = (time_step / count + offset) % 1
    return (1 - sampling_eps) * time_step + sampling_eps


def test_antithetic_times_match_the_released_time_distribution():
    generator = torch.Generator().manual_seed(7)
    ours = antithetic_uniform_times(16, generator=generator, sampling_eps=1e-3)
    reference = _released_time_distribution(16, seed=7, sampling_eps=1e-3)
    assert torch.equal(ours, reference)


def test_antithetic_times_stratify_the_unit_interval():
    times = antithetic_uniform_times(
        32, generator=torch.Generator().manual_seed(3), sampling_eps=1e-3
    )
    assert times.shape == (32,)
    assert float(times.min()) >= 1e-3
    assert float(times.max()) <= 1.0
    # One draw per stratum: sorting must land the i-th value in the i-th
    # 1/32-wide bin of [eps, 1].
    ordered = times.sort().values
    for index, value in enumerate(ordered.tolist()):
        low = 1e-3 + (1 - 1e-3) * index / 32
        high = 1e-3 + (1 - 1e-3) * (index + 1) / 32
        assert low <= value <= high


def test_per_sequence_sampling_draws_one_time_per_row():
    """Proved by RNG bookkeeping rather than by inspecting the mask.

    The objective draws its times, then one uniform per position. Replaying a
    twin generator through exactly ``batch`` times plus ``batch * length``
    position draws must leave it where the objective left its own, which is only
    true if the objective drew one time per sequence and not one per block.
    """
    decoder, batch = _fixture()
    ids = batch["input_ids"]
    rows, length = ids.shape
    blocks = math.ceil((length - 1) / decoder.config.block_width)

    used = torch.Generator().manual_seed(1234)
    decoder.diffusion_objective(
        ids,
        batch["precursor_mass"],
        batch["fingerprint"],
        isotope_ratios=batch["isotope_ratios"],
        generator=used,
        time_sampling="per_sequence_antithetic",
    )
    twin = torch.Generator().manual_seed(1234)
    torch.rand(rows, generator=twin)
    torch.rand((rows, length), generator=twin)
    assert torch.equal(
        torch.rand(4, generator=used), torch.rand(4, generator=twin)
    )

    used = torch.Generator().manual_seed(1234)
    decoder.diffusion_objective(
        ids,
        batch["precursor_mass"],
        batch["fingerprint"],
        isotope_ratios=batch["isotope_ratios"],
        generator=used,
        time_sampling="per_block_iid",
    )
    twin = torch.Generator().manual_seed(1234)
    torch.rand((rows, blocks), generator=twin)
    torch.rand((rows, length), generator=twin)
    assert torch.equal(
        torch.rand(4, generator=used), torch.rand(4, generator=twin)
    )


def test_per_sequence_sampling_changes_the_loss():
    decoder, batch = _fixture()
    assert _loss(
        decoder, batch, time_sampling="per_sequence_antithetic"
    ) == pytest.approx(FROZEN_ANTITHETIC_BLOCK_MEAN_LOSS, rel=1e-6)


def test_an_unknown_time_sampling_mode_is_refused():
    decoder, batch = _fixture()
    with pytest.raises(ValueError, match="unknown time_sampling"):
        _loss(decoder, batch, time_sampling="uniform")
    with pytest.raises(ValueError, match="time_sampling_eps"):
        _loss(decoder, batch, time_sampling_eps=0.0)


# --------------------------------------------------------------------------
# (d) the fp32 forward override
# --------------------------------------------------------------------------


def test_fp32_forward_context_is_a_no_op_when_off():
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with fp32_forward_context(False, "cpu"):
            assert torch.is_autocast_enabled("cpu")


def test_fp32_forward_context_disables_mixed_precision():
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        with fp32_forward_context(True, "cpu"):
            assert not torch.is_autocast_enabled("cpu")
        assert torch.is_autocast_enabled("cpu")


def test_fp32_forward_keeps_the_matmul_in_fp32():
    left = torch.randn(8, 8)
    right = torch.randn(8, 8)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with fp32_forward_context(False, "cpu"):
            assert (left @ right).dtype is torch.bfloat16
        with fp32_forward_context(True, "cpu"):
            assert (left @ right).dtype is torch.float32


# --------------------------------------------------------------------------
# (a) the learning-rate schedule
# --------------------------------------------------------------------------


def test_the_released_schedule_reproduces_its_own_terminal_rate():
    """Guards the derivation's input against a transcription error."""
    terminal = released_learning_rate(RELEASED_WARMUP_STEPS + RELEASED_COSINE_LAST_STEP)
    assert terminal == pytest.approx(RELEASED_TERMINAL_LEARNING_RATE, rel=1e-9)


def test_the_final_decade_is_a_decade():
    first, last = released_final_decade_steps()
    assert released_learning_rate(first) == pytest.approx(
        10 * RELEASED_TERMINAL_LEARNING_RATE, rel=1e-4
    )
    assert released_learning_rate(first - 1) > 10 * RELEASED_TERMINAL_LEARNING_RATE
    assert last - first == 14890


def test_the_derived_peak_spends_the_derived_budget():
    budget = released_final_decade_displacement()
    peak = derive_peak_learning_rate(
        total_steps=20000,
        warmup_steps=ADAMW_SECOND_MOMENT_WINDOW,
        floor=RELEASED_TERMINAL_LEARNING_RATE,
    )
    assert peak == pytest.approx(3.1647e-7, rel=1e-3)
    assert peak / RELEASED_TERMINAL_LEARNING_RATE == pytest.approx(6.0, rel=1e-2)
    spent = schedule_displacement(
        peak,
        total_steps=20000,
        warmup_steps=ADAMW_SECOND_MOMENT_WINDOW,
        floor=RELEASED_TERMINAL_LEARNING_RATE,
    )
    assert spent == pytest.approx(budget, rel=1e-9)


def test_the_current_recipe_blows_the_budget():
    """The number that makes the case: 1e-5 flat for 20,000 steps."""
    budget = released_final_decade_displacement()
    assert 1e-5 * 20000 / budget == pytest.approx(54.6, rel=1e-2)
    assert 1e-5 * 100000 / budget == pytest.approx(272.8, rel=1e-2)


def test_a_run_too_long_for_its_budget_is_refused():
    with pytest.raises(ValueError, match="shorten the run"):
        derive_peak_learning_rate(total_steps=10**7)


def test_the_schedule_shape():
    factors = [
        warmup_cosine_factor(
            step, total_steps=1000, warmup_steps=100, floor_factor=0.1
        )
        for step in range(1000)
    ]
    assert factors[0] == pytest.approx(1e-6, abs=1e-9)
    assert factors[100] == pytest.approx(1.0)
    assert factors[-1] == pytest.approx(0.1, abs=2e-5)
    assert factors[:100] == sorted(factors[:100])
    assert factors[100:] == sorted(factors[100:], reverse=True)


def test_the_scheduler_moves_the_optimizer_rate():
    peak = 3.2e-7
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=peak)
    scheduler = build_warmup_cosine_scheduler(
        optimizer, total_steps=1000, warmup_steps=100, floor=RELEASED_TERMINAL_LEARNING_RATE
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(peak * 1e-6, rel=1e-6)
    seen = []
    for _ in range(1000):
        seen.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    assert max(seen) == pytest.approx(peak, rel=1e-6)
    assert seen[-1] == pytest.approx(RELEASED_TERMINAL_LEARNING_RATE, rel=1e-3)


def test_the_module_wires_the_schedule_and_refuses_a_bad_one():
    config = _fixture()[0].config
    module = MarlinLightningModule(
        config,
        learning_rate=3.2e-7,
        lr_schedule="warmup_cosine",
        lr_warmup_steps=10,
        lr_total_steps=100,
        lr_min=RELEASED_TERMINAL_LEARNING_RATE,
    )
    wired = module.configure_optimizers()
    assert wired["lr_scheduler"]["interval"] == "step"
    assert wired["optimizer"].param_groups[0]["lr"] == pytest.approx(
        3.2e-7 * 1e-6, rel=1e-6
    )
    with pytest.raises(ValueError, match="lr_total_steps"):
        MarlinLightningModule(config, lr_schedule="warmup_cosine")
    with pytest.raises(ValueError, match="lr_min"):
        MarlinLightningModule(
            config,
            learning_rate=1e-9,
            lr_schedule="warmup_cosine",
            lr_warmup_steps=10,
            lr_total_steps=100,
            lr_min=1e-7,
        )


# --------------------------------------------------------------------------
# Bounding the replay problem: early stopping on the held-out probe.
# --------------------------------------------------------------------------


class _FakeTrainer:
    is_global_zero = True
    should_stop = False


def test_the_probe_metric_may_drive_selection():
    assert "probe_top1_predicted" in PROBE_SELECTION_METRICS
    assert "probe_top1_predicted" not in PANEL_SELECTION_METRICS
    selector = CheckpointSelector(metric="probe_top1_predicted", patience=2)
    assert selector.update(200, {"probe_top1_predicted": 0.51}).improved
    assert selector.update(400, {"probe_top1_predicted": 0.57}).improved
    assert not selector.update(600, {"probe_top1_predicted": 0.55}).should_stop
    decision = selector.update(800, {"probe_top1_predicted": 0.54})
    assert decision.should_stop
    assert decision.best_step == 400
    assert decision.best_value == pytest.approx(0.57)


def test_the_probe_stops_the_trainer_when_it_stops_improving(tmp_path):
    from marlin.conditioning_probe import PeriodicConditioningProbe

    selection_path = tmp_path / "selection" / "probe_selection.json"
    probe = PeriodicConditioningProbe(
        [{"input_ids": torch.zeros(2, 2, dtype=torch.long)}],
        interval_steps=200,
        selector=CheckpointSelector(metric="probe_top1_predicted", patience=1),
        selection_path=selection_path,
    )
    trainer = _FakeTrainer()
    probe._select(trainer, 200, {"probe_top1_predicted": 0.6})
    assert not trainer.should_stop
    probe._select(trainer, 400, {"probe_top1_predicted": 0.4})
    assert trainer.should_stop
    assert selection_path.exists()
    assert "probe_top1_predicted" in selection_path.read_text()


def test_a_probe_without_a_selector_never_stops_a_run():
    from marlin.conditioning_probe import PeriodicConditioningProbe

    probe = PeriodicConditioningProbe(
        [{"input_ids": torch.zeros(2, 2, dtype=torch.long)}],
        interval_steps=200,
    )
    trainer = _FakeTrainer()
    probe._select(trainer, 200, {"probe_top1_predicted": 0.6})
    probe._select(trainer, 400, {"probe_top1_predicted": 0.1})
    assert not trainer.should_stop
