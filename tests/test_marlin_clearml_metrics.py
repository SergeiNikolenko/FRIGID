from types import SimpleNamespace

import pytest
import torch

from marlin.clearml_metrics import ClearMLTrainingMetrics


class _Logger:
    def __init__(self) -> None:
        self.calls = []

    def report_scalar(self, **kwargs) -> None:
        self.calls.append(kwargs)


def _reporter() -> tuple[ClearMLTrainingMetrics, _Logger]:
    logger = _Logger()
    task = SimpleNamespace(get_logger=lambda: logger)
    return ClearMLTrainingMetrics(task), logger


def test_clearml_training_metrics_reports_once_per_optimizer_step() -> None:
    reporter, logger = _reporter()
    trainer = SimpleNamespace(
        is_global_zero=True,
        global_step=1,
        callback_metrics={
            "train_loss": torch.tensor(2.5),
            "train_top1_accuracy_step": torch.tensor(0.25),
            "ignored": torch.tensor(99.0),
        },
        logged_metrics={"learning_rate": 5e-5},
    )

    reporter.on_train_batch_end(trainer, None, None, None, 0)
    reporter.on_train_batch_end(trainer, None, None, None, 1)

    assert {(call["series"], call["iteration"]) for call in logger.calls} == {
        ("learning_rate", 1),
        ("train_loss", 1),
        ("train_top1_accuracy", 1),
    }
    assert all(call["title"] == "Training metrics" for call in logger.calls)


def test_clearml_training_metrics_only_reports_global_zero_finite_scalars() -> None:
    reporter, logger = _reporter()
    trainer = SimpleNamespace(
        is_global_zero=False,
        global_step=1,
        callback_metrics={"train_loss": torch.tensor(1.0)},
        logged_metrics={},
    )
    reporter.on_train_batch_end(trainer, None, None, None, 0)
    trainer.is_global_zero = True
    trainer.global_step = 2
    trainer.callback_metrics = {
        "train_loss": torch.tensor(float("nan")),
        "train_vector": torch.tensor([1.0, 2.0]),
        "fingerprint_noise_fraction": torch.tensor(0.2),
    }
    reporter.on_train_batch_end(trainer, None, None, None, 1)

    assert logger.calls == [
        {
            "title": "Training metrics",
            "series": "fingerprint_noise_fraction",
            "value": pytest.approx(0.2),
            "iteration": 2,
        }
    ]


def test_clearml_training_metrics_does_not_repeat_sparse_metrics() -> None:
    reporter, logger = _reporter()
    trainer = SimpleNamespace(
        is_global_zero=True,
        global_step=51,
        callback_metrics={
            "train_loss": torch.tensor(2.0),
            "train_masked_token_accuracy_top1": torch.tensor(0.4),
        },
        logged_metrics={},
    )
    module = SimpleNamespace(_last_metric_step=50)

    reporter.on_train_batch_end(trainer, module, None, None, 0)

    assert [(call["series"], call["iteration"]) for call in logger.calls] == [
        ("train_loss", 51)
    ]

    trainer.global_step = 100
    module._last_metric_step = 100
    reporter.on_train_batch_end(trainer, module, None, None, 1)

    assert [(call["series"], call["iteration"]) for call in logger.calls[-2:]] == [
        ("train_loss", 100),
        ("train_masked_token_accuracy_top1", 100),
    ]
