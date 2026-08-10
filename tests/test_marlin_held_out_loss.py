import pytest
import torch

from marlin.model import MarlinDecoderConfig
from marlin.training import (
    MarlinLightningModule,
    PeriodicHeldOutLoss,
    holdout_split_report,
    structure_disjoint_holdout,
)


def _rows(keys):
    return [{"inchikey_first_block": key, "safe": f"c1ccccc1.{index}"}
            for index, key in enumerate(keys)]


def test_zero_fraction_holds_nothing_out():
    rows = _rows(["AAA", "BBB", "CCC"])

    train, holdout = structure_disjoint_holdout(rows, fraction=0.0)

    assert train == [0, 1, 2]
    assert holdout == []


def test_holdout_is_taken_on_the_structure_not_the_row():
    """Two spectra of one molecule must land on the same side, otherwise the
    optimizer sees the structure whose loss is reported as held out."""
    rows = _rows(["AAA", "BBB", "AAA", "CCC", "BBB", "DDD", "EEE", "FFF"])

    train, holdout = structure_disjoint_holdout(rows, fraction=0.4, seed=3)
    report = holdout_split_report(rows, train, holdout)

    holdout_keys = {rows[index]["inchikey_first_block"] for index in holdout}
    train_keys = {rows[index]["inchikey_first_block"] for index in train}
    assert holdout_keys
    assert not holdout_keys & train_keys
    assert report["shared_structures"] == 0
    assert report["holdout_rows"] == len(holdout)
    assert report["training_structures"] + report["holdout_structures"] == 6


def test_holdout_is_stable_under_row_order_and_new_rows():
    """A hash of the connectivity block, not a shuffle: adding spectra must not
    move an existing structure across the boundary."""
    keys = [f"KEY{index:03d}" for index in range(60)]
    rows = _rows(keys)
    _, holdout = structure_disjoint_holdout(rows, fraction=0.2, seed=7)
    held = {rows[index]["inchikey_first_block"] for index in holdout}

    shuffled = _rows(list(reversed(keys)))
    _, shuffled_holdout = structure_disjoint_holdout(shuffled, fraction=0.2, seed=7)
    grown = _rows(keys + [f"NEW{index:03d}" for index in range(20)])
    _, grown_holdout = structure_disjoint_holdout(grown, fraction=0.2, seed=7)

    assert {shuffled[index]["inchikey_first_block"] for index in shuffled_holdout} == held
    assert held <= {grown[index]["inchikey_first_block"] for index in grown_holdout}


def test_holdout_split_rejects_impossible_configurations():
    rows = _rows(["AAA", "BBB"])
    with pytest.raises(ValueError, match=r"must be in \[0, 1\)"):
        structure_disjoint_holdout(rows, fraction=1.0)
    with pytest.raises(ValueError, match="selected no structures"):
        structure_disjoint_holdout(rows, fraction=1e-9)
    with pytest.raises(ValueError, match="has no inchikey_first_block"):
        structure_disjoint_holdout([{"safe": "C"}], fraction=0.1)


def test_holdout_split_report_refuses_a_leaking_split():
    rows = _rows(["AAA", "AAA"])

    with pytest.raises(ValueError, match="shares structures with training"):
        holdout_split_report(rows, [0], [1])


def _tiny_module():
    config = MarlinDecoderConfig(
        vocab_size=16,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=6,
        block_width=2,
        fingerprint_bits=8,
        dropout=0.0,
        mask_token_id=3,
        pad_token_id=0,
    )
    return MarlinLightningModule(config, ema_decay=0.5)


def _batch():
    fingerprint = torch.zeros((2, 8))
    fingerprint[0, [1, 4]] = 1.0
    fingerprint[1, [2]] = 1.0
    return {
        "input_ids": torch.tensor([[1, 5, 6, 2], [1, 7, 8, 2]]),
        "precursor_mass": torch.tensor([120.0, 140.0]),
        "fingerprint": fingerprint,
        "isotope_ratios": torch.zeros((2, 2)),
    }


def test_held_out_loss_is_deterministic_across_passes():
    """The generator is re-seeded per pass, so two checkpoints are scored on the
    same mask draws and only the weights differ."""
    module = _tiny_module()
    callback = PeriodicHeldOutLoss([_batch(), _batch()], interval_steps=10, seed=5)

    first = callback.evaluate(module)
    second = callback.evaluate(module)

    assert first["loss"] == pytest.approx(second["loss"])
    assert "masked_token_accuracy_top1" in first
    assert module.training


def test_held_out_loss_does_not_corrupt_the_fingerprint():
    """Symmetric corruption is a training augmentation; inference conditions on
    the uncorrupted predicted fingerprint."""
    module = _tiny_module()
    batch = _batch()
    seen = []

    def record(*args, **kwargs):
        seen.append(args[2].clone())
        return torch.tensor(1.0), {}

    module.decoder.diffusion_objective = record
    PeriodicHeldOutLoss([batch], interval_steps=10).evaluate(module)

    assert torch.equal(seen[0], batch["fingerprint"])


class _StubModule:
    def __init__(self):
        self.training = True
        self.device = torch.device("cpu")
        self.logged = {}
        self.eos_loss_weight = 1.0
        self.eos_mask_probability = 0.0
        self.balanced_token_loss_alpha = 0.0
        self.token_loss_weight_max = 20.0
        self.full_sequence_mask_probability = 0.0
        self.calls = 0
        module = self

        class _Decoder:
            def diffusion_objective(self, *args, **kwargs):
                module.calls += 1
                return torch.tensor(0.5), {"mask_fraction": torch.tensor(0.25)}

        self.decoder = _Decoder()

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def log(self, name, value, **kwargs):
        self.logged[name] = value


class _Logger:
    def __init__(self):
        self.scalars = []

    def report_scalar(self, **kwargs):
        self.scalars.append(kwargs)


def test_held_out_loss_runs_only_on_the_evaluation_interval():
    logger = _Logger()
    task = type("Task", (), {"get_logger": lambda self: logger})()
    module = _StubModule()
    callback = PeriodicHeldOutLoss(
        [_batch()], interval_steps=100, clearml_task=task
    )
    trainer = type("Trainer", (), {"global_step": 0, "is_global_zero": True})()

    for step in (0, 50, 100, 100, 150, 200):
        trainer.global_step = step
        callback.on_train_batch_end(trainer, module, None, None, 0)

    assert module.calls == 2
    assert module.logged["val_loss"] == pytest.approx(0.5)
    assert module.logged["val_mask_fraction"] == pytest.approx(0.25)
    assert [entry["iteration"] for entry in logger.scalars] == [100, 100, 200, 200]
    assert {entry["series"] for entry in logger.scalars} == {
        "val_loss",
        "val_mask_fraction",
    }


def test_held_out_loss_averages_over_batches_and_rejects_an_empty_loader():
    module = _StubModule()
    values = PeriodicHeldOutLoss(
        [_batch(), _batch(), _batch()], interval_steps=10, max_batches=2
    ).evaluate(module)

    assert module.calls == 2
    assert values["loss"] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="produced no batches"):
        PeriodicHeldOutLoss([], interval_steps=10).evaluate(_StubModule())


def test_held_out_loss_rejects_invalid_configuration():
    with pytest.raises(ValueError, match="interval must be positive"):
        PeriodicHeldOutLoss([], interval_steps=0)
    with pytest.raises(ValueError, match="max_batches must be positive"):
        PeriodicHeldOutLoss([], interval_steps=10, max_batches=0)
