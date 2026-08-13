import pytest
import torch

from marlin.conditioning_probe import (
    PeriodicConditioningProbe,
    build_probe_batches,
    conditioning_probe_metrics,
    select_probe_rows,
)
from marlin.model import MarlinDecoderConfig, MarlinDecoder


VOCAB = 16
PAD = 0
MASK = 3


class _Config:
    vocab_size = VOCAB
    pad_token_id = PAD
    mask_token_id = MASK


class _ScriptedDecoder(torch.nn.Module):
    """A decoder whose only skill is reading the fingerprint.

    ``gold_bit`` is the fingerprint bit that identifies each row's answer. With
    ``sensitivity=1`` the model puts all its mass on the gold token when the
    conditioning vector carries that bit and spreads it uniformly otherwise, so
    the probe's content and presence terms have known signs. With
    ``sensitivity=0`` the model ignores the fingerprint entirely, which is the
    behaviour ``probe_conditioning_free_loss_fraction`` must read as 1.0.
    """

    def __init__(self, gold_bits, *, sensitivity: float = 1.0):
        super().__init__()
        self.config = _Config()
        self.gold_bits = gold_bits
        self.sensitivity = sensitivity
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def two_stream_logits(
        self,
        clean_ids,
        noised_ids,
        precursor_mass,
        fingerprint,
        isotope_ratios=None,
        *,
        include_mass_conditioning=True,
    ):
        del noised_ids, precursor_mass, isotope_ratios, include_mass_conditioning
        batch, length = clean_ids.shape
        logits = torch.zeros((batch, length, VOCAB))
        for row in range(batch):
            carried = float(fingerprint[row, self.gold_bits[row]])
            boost = 8.0 * self.sensitivity * carried
            logits[row].scatter_(
                1, clean_ids[row].unsqueeze(-1), torch.full((length, 1), boost)
            )
        return logits


def _batch():
    true = torch.zeros((3, 8))
    predicted = torch.zeros((3, 8))
    for row, bit in enumerate((1, 2, 5)):
        true[row, bit] = 1.0
    # The predicted fingerprint carries the identifying bit for one row only.
    predicted[0, 1] = 1.0
    predicted[1, 7] = 1.0
    predicted[2, 6] = 1.0
    return {
        "input_ids": torch.tensor(
            [[1, 5, 6, 7, 2], [1, 7, 8, 9, 2], [1, 4, 5, 6, 2]]
        ),
        "precursor_mass": torch.tensor([120.0, 140.0, 150.0]),
        "fingerprint": predicted,
        "true_fingerprint": true,
        "isotope_ratios": torch.zeros((3, 2)),
    }


def test_a_content_blind_model_is_reported_as_conditioning_free():
    """The metric that would have caught "the decoder reads presence, not
    content": every row's loss is explained by the conditioning-free prior."""
    decoder = _ScriptedDecoder([1, 2, 5], sensitivity=0.0)

    values = conditioning_probe_metrics(decoder, [_batch()], seed=1)

    assert values["probe_conditioning_free_loss_fraction"] == pytest.approx(1.0)
    assert values["probe_conditioning_free_loss_fraction_predicted"] == pytest.approx(
        1.0
    )
    assert values["probe_conditioning_content_gain"] == pytest.approx(0.0, abs=1e-6)
    assert values["probe_conditioning_presence_gain"] == pytest.approx(0.0, abs=1e-6)
    assert values["probe_top1_gap_true_minus_predicted"] == pytest.approx(0.0)


def test_a_content_reading_model_shows_a_positive_content_gain():
    decoder = _ScriptedDecoder([1, 2, 5], sensitivity=1.0)

    values = conditioning_probe_metrics(decoder, [_batch()], seed=1)

    assert values["probe_conditioning_content_gain"] > 0.0
    assert values["probe_conditioning_free_loss_fraction"] == pytest.approx(0.0)
    assert values["probe_top1_true"] == pytest.approx(1.0)
    # Mismatched and absent conditioning are equally useless to this model, so
    # the presence term is zero: content is all it reads.
    assert values["probe_conditioning_presence_gain"] == pytest.approx(0.0, abs=1e-6)


def test_the_probe_measures_the_true_versus_predicted_fingerprint_gap():
    """The direct probe of the 0.750 vs 0.547 teacher-forced gap: the true
    fingerprint identifies every row, the predicted one identifies row 0."""
    decoder = _ScriptedDecoder([1, 2, 5], sensitivity=1.0)

    values = conditioning_probe_metrics(decoder, [_batch()], seed=1)

    assert values["probe_top1_true"] == pytest.approx(1.0)
    assert 0.0 < values["probe_top1_predicted"] < 1.0
    assert values["probe_top1_gap_true_minus_predicted"] == pytest.approx(
        values["probe_top1_true"] - values["probe_top1_predicted"]
    )
    assert values["probe_conditioning_free_loss_fraction_predicted"] > 0.0


def test_the_probe_is_deterministic_and_leaves_the_module_training():
    """Two checkpoints must be scored on the same mask draws, and probing must
    not silently leave the decoder in eval mode."""
    torch.manual_seed(0)
    config = MarlinDecoderConfig(
        vocab_size=VOCAB,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=8,
        block_width=2,
        fingerprint_bits=8,
        dropout=0.5,
        mask_token_id=MASK,
        pad_token_id=PAD,
    )
    decoder = MarlinDecoder(config)
    decoder.train()

    first = conditioning_probe_metrics(decoder, [_batch()], seed=7)
    second = conditioning_probe_metrics(decoder, [_batch()], seed=7)
    other = conditioning_probe_metrics(decoder, [_batch()], seed=8)

    assert first == second
    assert first != other
    assert decoder.training


def test_every_probe_row_is_scored_even_when_it_draws_no_mask():
    decoder = _ScriptedDecoder([1, 2, 5], sensitivity=1.0)

    values = conditioning_probe_metrics(
        decoder, [_batch()], seed=1, mask_probability=1e-9
    )

    assert values["probe_rows"] == 3.0
    assert values["probe_masked_tokens"] == 3.0


def test_the_probe_refuses_configurations_that_would_report_a_false_zero():
    decoder = _ScriptedDecoder([1], sensitivity=1.0)
    single = {
        "input_ids": torch.tensor([[1, 5, 2]]),
        "precursor_mass": torch.tensor([120.0]),
        "fingerprint": torch.zeros((1, 8)),
        "true_fingerprint": torch.zeros((1, 8)),
        "isotope_ratios": torch.zeros((1, 2)),
    }

    # Rolling a one-row batch is the identity, so the content gain would be 0.
    with pytest.raises(ValueError, match="at least two rows"):
        conditioning_probe_metrics(decoder, [single], seed=1)
    with pytest.raises(ValueError, match="received no batches"):
        conditioning_probe_metrics(decoder, [], seed=1)
    with pytest.raises(ValueError, match="mask probability must be in"):
        conditioning_probe_metrics(decoder, [_batch()], mask_probability=0.0)
    with pytest.raises(ValueError, match="margin must be non-negative"):
        conditioning_probe_metrics(
            decoder, [_batch()], conditioning_free_margin=-1.0
        )


def test_probe_selection_depends_on_the_names_and_nothing_else():
    """A hash order, not a shuffle or a head slice: the same spectra must be
    probed whatever order the metadata happens to list them in, and dropping a
    spectrum the probe did not pick must leave the probe identical."""
    rows = [{"spec_name": f"S{index:03d}"} for index in range(40)]

    chosen = select_probe_rows(rows, size=8, seed=2)
    names = {rows[index]["spec_name"] for index in chosen}

    reversed_rows = list(reversed(rows))
    reversed_names = {
        reversed_rows[index]["spec_name"]
        for index in select_probe_rows(reversed_rows, size=8, seed=2)
    }
    unpicked = next(row for row in rows if row["spec_name"] not in names)
    pruned = [row for row in rows if row is not unpicked]
    pruned_names = {
        pruned[index]["spec_name"]
        for index in select_probe_rows(pruned, size=8, seed=2)
    }

    assert len(chosen) == 8
    assert chosen == sorted(chosen)
    assert reversed_names == names
    assert pruned_names == names
    assert select_probe_rows(rows, size=8, seed=3) != chosen


def test_probe_selection_refuses_an_empty_or_zero_sized_request():
    with pytest.raises(ValueError, match="probe size must be positive"):
        select_probe_rows([{"spec_name": "A"}], size=0)
    with pytest.raises(ValueError, match="at least one row"):
        select_probe_rows([], size=4)


class _Dataset:
    def __init__(self):
        self.rows = [
            {
                "spec_name": f"S{index}",
                "safe": f"safe-{index}",
                "fingerprint": torch.full((8,), float(index)).numpy(),
            }
            for index in range(6)
        ]


class _Collator:
    """Stand in for MarlinCollator: a missing fingerprint means the gold one."""

    def __call__(self, examples):
        size = len(examples)
        provided = [example.get("fingerprint") for example in examples]
        if all(value is None for value in provided):
            fingerprint = torch.full((size, 8), -1.0)
        else:
            fingerprint = torch.stack([torch.as_tensor(value) for value in provided])
        return {
            "input_ids": torch.tensor([[1, 5, 2]] * size),
            "fingerprint": fingerprint,
            "precursor_mass": torch.zeros(size),
            "isotope_ratios": torch.zeros((size, 2)),
        }


def test_build_probe_batches_carries_both_fingerprints():
    batches = build_probe_batches(
        _Dataset(), _Collator(), size=4, batch_size=2, seed=1
    )

    assert len(batches) == 2
    for batch in batches:
        assert batch["fingerprint"].shape == (2, 8)
        # The stripped collation is the gold fingerprint, and it is kept apart
        # from the predicted one rather than overwriting it.
        assert torch.equal(batch["true_fingerprint"], torch.full((2, 8), -1.0))
        assert not torch.equal(batch["fingerprint"], batch["true_fingerprint"])


def test_build_probe_batches_refuses_a_batch_that_cannot_be_rolled():
    with pytest.raises(ValueError, match="at least 2"):
        build_probe_batches(_Dataset(), _Collator(), size=4, batch_size=1)


class _StubModule:
    def __init__(self):
        self.decoder = _ScriptedDecoder([1, 2, 5], sensitivity=1.0)
        self.logged: dict[str, float] = {}

    def log(self, name, value, **kwargs):
        self.logged[name] = float(value)


class _Logger:
    def __init__(self):
        self.scalars = []

    def report_scalar(self, **kwargs):
        self.scalars.append(kwargs)


def test_periodic_probe_runs_only_on_its_interval():
    logger = _Logger()
    task = type("Task", (), {"get_logger": lambda self: logger})()
    module = _StubModule()
    callback = PeriodicConditioningProbe(
        [_batch()], interval_steps=100, seed=1, clearml_task=task
    )
    trainer = type("Trainer", (), {"global_step": 0, "is_global_zero": True})()

    for step in (0, 50, 100, 100, 150, 200):
        trainer.global_step = step
        callback.on_train_batch_end(trainer, module, None, None, 0)

    assert sorted({entry["iteration"] for entry in logger.scalars}) == [100, 200]
    assert "probe_top1_true" in module.logged
    assert "probe_conditioning_free_loss_fraction" in module.logged


def test_periodic_probe_refuses_an_impossible_configuration():
    with pytest.raises(ValueError, match="interval must be positive"):
        PeriodicConditioningProbe([_batch()], interval_steps=0)
    with pytest.raises(ValueError, match="at least one batch"):
        PeriodicConditioningProbe([], interval_steps=10)
