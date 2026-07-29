"""Explicit ClearML reporting for MARLIN training metrics."""

from __future__ import annotations

import math
from typing import Any

import lightning as L
import torch


_TRAINING_METRICS = {
    "adaptation_stage",
    "fingerprint_noise_fraction",
    "grad_norm",
    "learning_rate",
    "micro_batch_size",
    "train_loss",
    "trainable_parameter_fraction",
}
_SPARSE_TRAINING_METRICS = {
    "adaptation_stage",
    "grad_norm",
    "trainable_parameter_fraction",
}


def _metric_name(name: str) -> str | None:
    normalized = name.removesuffix("_step")
    if normalized in _TRAINING_METRICS or normalized.startswith("train_"):
        return normalized
    return None


def _scalar(value: Any) -> float | None:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().float().cpu().item()
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


class ClearMLTrainingMetrics(L.Callback):
    """Publish Lightning step metrics to the active ClearML task.

    Lightning falls back to ``CSVLogger`` when TensorBoard is absent, so
    ClearML's TensorBoard auto-connect cannot see ``LightningModule.log``.
    This callback makes the tracking contract explicit and independent of the
    optional TensorBoard package.
    """

    def __init__(self, task=None) -> None:
        super().__init__()
        self.logger = task.get_logger() if task is not None else None
        self._last_step = -1

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        del outputs, batch, batch_idx
        if self.logger is None or not trainer.is_global_zero:
            return
        step = int(trainer.global_step)
        if step <= 0 or step == self._last_step:
            return
        self._last_step = step

        metrics = dict(trainer.callback_metrics)
        metrics.update(trainer.logged_metrics)
        reported: set[str] = set()
        for raw_name, raw_value in sorted(metrics.items()):
            name = _metric_name(str(raw_name))
            value = _scalar(raw_value)
            if name is None or value is None or name in reported:
                continue
            is_sparse = (
                name in _SPARSE_TRAINING_METRICS
                or (name.startswith("train_") and name != "train_loss")
            )
            if (
                is_sparse
                and pl_module is not None
                and int(getattr(pl_module, "_last_metric_step", -1)) != step
            ):
                continue
            self.logger.report_scalar(
                title="Training metrics",
                series=name,
                value=value,
                iteration=step,
            )
            reported.add(name)
