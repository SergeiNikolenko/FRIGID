"""Per-pathway gradient, clipping and EMA telemetry for MARLIN adaptation.

The adaptation recipe logged exactly one gradient number: a global norm over
every parameter, every fiftieth step
(``src/marlin/training.py:775-785``). That cannot answer the question the
architecture poses. The decoder is a backbone plus a conditioning pathway -- the
fingerprint conditioner and the cross-attention stack that injects it -- and our
own diagnosis says the decoder reacts to the *presence* of a conditioning vector
far more than to its *content*. A starved conditioner would be invisible in a
global norm dominated by 250M backbone parameters.

Everything here is opt-in. Nothing in this module runs unless a caller attaches
one of the callbacks.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import lightning as L
import torch


PARAMETER_GROUPS = (
    "fingerprint_conditioner",
    "mass_conditioner",
    "isotope_conditioner",
    "cross_attention",
    "embedding",
    "head",
    "backbone",
)


def parameter_group(name: str) -> str:
    """Map a decoder parameter name onto the pathway it belongs to.

    The split follows the architecture rather than the module tree. ``norm2``
    joins the cross-attention group because it is the normalisation that the
    cross-attention residual owns, and because
    ``MarlinLightningModule._stage_parameter_is_trainable`` unfreezes the two
    together -- grouping them the same way keeps the gradient curves readable
    across a staged-unfreezing boundary.
    """
    local = name.removeprefix("decoder.")
    if local.startswith("conditioner.fingerprint."):
        return "fingerprint_conditioner"
    if local.startswith("conditioner.mass."):
        return "mass_conditioner"
    if local.startswith("conditioner.isotope."):
        return "isotope_conditioner"
    if ".cross_attention." in local or ".norm2." in local:
        return "cross_attention"
    if local.startswith(("token_embedding.", "position_embedding.")):
        return "embedding"
    if local == "output_bias" or local.startswith(
        ("prediction_dense.", "prediction_norm.")
    ):
        return "head"
    return "backbone"


def gradient_group_norms(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> dict[str, float]:
    """Return pre-clip gradient statistics per pathway, plus the global norm.

    Three readings per group, because one does not answer the question:

    * ``grad_norm/<group>`` is the plain L2 norm of that group's gradient;
    * ``grad_rms/<group>`` divides it by the square root of the element count,
      so a 4096-bit fingerprint embedding and a LayerNorm are on one scale;
    * ``grad_share/<group>`` is the group's share of the squared global norm,
      which is what "the conditioning pathway is starved" literally means.

    ``grad_covered/<group>`` is the fraction of the group's elements that
    carried a gradient at all. A frozen stage reads 0 there, which distinguishes
    "this pathway received nothing" from "this pathway received zero".
    """
    squares: dict[str, float] = {group: 0.0 for group in PARAMETER_GROUPS}
    counts: dict[str, int] = {group: 0 for group in PARAMETER_GROUPS}
    totals: dict[str, int] = {group: 0 for group in PARAMETER_GROUPS}
    for name, parameter in named_parameters:
        group = parameter_group(name)
        elements = parameter.numel()
        totals[group] += elements
        if parameter.grad is None:
            continue
        counts[group] += elements
        squares[group] += float(parameter.grad.detach().float().pow(2).sum())

    global_square = sum(squares.values())
    metrics: dict[str, float] = {
        "grad_norm/global": math.sqrt(global_square),
        "grad_parameters_with_gradient": float(sum(counts.values())),
    }
    for group in PARAMETER_GROUPS:
        if not totals[group]:
            # A configuration without this pathway at all: report nothing rather
            # than a zero that would read as a starved gradient.
            continue
        norm = math.sqrt(squares[group])
        metrics[f"grad_norm/{group}"] = norm
        metrics[f"grad_rms/{group}"] = (
            norm / math.sqrt(counts[group]) if counts[group] else 0.0
        )
        metrics[f"grad_share/{group}"] = (
            squares[group] / global_square if global_square > 0.0 else 0.0
        )
        metrics[f"grad_covered/{group}"] = counts[group] / totals[group]
    return metrics


def ema_divergence(
    shadow_params: Sequence[torch.Tensor],
    parameters: Iterable[torch.nn.Parameter],
) -> dict[str, float]:
    """Distance between the EMA shadow and the live weights.

    Nothing in the recipe ever logged this, so "did the EMA diverge from the
    live weights?" was unanswerable from a finished run even though the
    evaluator can be pointed at either. The relative distance is the readable
    one: it is scale free, so it is comparable between a warm start and step
    100,000.
    """
    parameter_list = list(parameters)
    if len(parameter_list) != len(shadow_params):
        raise ValueError(
            "EMA shadow count does not match the parameters: "
            f"{len(shadow_params)} != {len(parameter_list)}"
        )
    difference = 0.0
    live = 0.0
    with torch.no_grad():
        for shadow, parameter in zip(shadow_params, parameter_list):
            value = parameter.detach().to(device=shadow.device, dtype=torch.float32)
            difference += float((shadow.float() - value).pow(2).sum())
            live += float(value.pow(2).sum())
    return {
        "ema_distance": math.sqrt(difference),
        "ema_parameter_norm": math.sqrt(live),
        "ema_relative_distance": (
            math.sqrt(difference / live) if live > 0.0 else 0.0
        ),
    }


class _PeriodicReporter(L.Callback):
    """Shared interval, ClearML reporting and Lightning logging."""

    title = "Diagnostics"

    def __init__(
        self,
        *,
        interval_steps: int,
        clearml_task=None,
    ) -> None:
        super().__init__()
        if interval_steps <= 0:
            raise ValueError("diagnostic interval must be positive")
        self.interval_steps = interval_steps
        self.clearml_task = clearml_task
        self._last_reported_step = -1

    def _due(self, step: int) -> bool:
        return step % self.interval_steps == 0 and step != self._last_reported_step

    def _publish(self, pl_module, step: int, values: dict[str, float]) -> None:
        self._last_reported_step = step
        for name, value in values.items():
            if not math.isfinite(value):
                continue
            pl_module.log(name, value, on_step=True, sync_dist=False)
        if self.clearml_task is None:
            return
        logger = self.clearml_task.get_logger()
        for name, value in values.items():
            if not math.isfinite(value):
                continue
            logger.report_scalar(
                title=self.title,
                series=name,
                value=value,
                iteration=step,
            )


class GradientDiagnostics(_PeriodicReporter):
    """Log pre-clip gradient norms per pathway, and the true clipping rate.

    Lightning calls ``on_before_optimizer_step`` immediately before
    ``_clip_gradients``
    (``lightning/pytorch/plugins/precision/precision.py:87-92``), so the norms
    seen here are the ones the clip threshold is compared against.

    The global norm is recomputed on **every** optimizer step, not only on
    reporting steps, because a clipping rate measured on every fiftieth step is
    a subsample and not a rate. Only the reporting is periodic.
    """

    title = "Gradient diagnostics"

    def __init__(
        self,
        *,
        interval_steps: int,
        clip_value: float | None = None,
        clearml_task=None,
    ) -> None:
        super().__init__(interval_steps=interval_steps, clearml_task=clearml_task)
        if clip_value is not None and clip_value <= 0:
            raise ValueError("clip_value must be positive when given")
        self.clip_value = clip_value
        self.optimizer_steps = 0
        self.clipped_steps = 0

    def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
        del optimizer
        values = gradient_group_norms(pl_module.named_parameters())
        self.optimizer_steps += 1
        global_norm = values["grad_norm/global"]
        if self.clip_value is not None:
            clipped = global_norm > self.clip_value
            self.clipped_steps += int(clipped)
            values["grad_clipped"] = float(clipped)
            values["grad_clip_rate"] = self.clipped_steps / self.optimizer_steps
            # How much of the update survives the clip. 1.0 means untouched.
            values["grad_clip_scale"] = (
                min(1.0, self.clip_value / global_norm) if global_norm > 0.0 else 1.0
            )
        step = int(trainer.global_step)
        if not self._due(step):
            return
        self._publish(pl_module, step, values)


class EmaDivergence(_PeriodicReporter):
    """Log how far the EMA shadow has drifted from the live weights."""

    title = "EMA divergence"

    def on_train_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        del outputs, batch, batch_idx
        step = int(trainer.global_step)
        if step <= 0 or not self._due(step):
            return
        values = ema_divergence(
            pl_module.ema.shadow_params, pl_module.decoder.parameters()
        )
        self._publish(pl_module, step, values)
