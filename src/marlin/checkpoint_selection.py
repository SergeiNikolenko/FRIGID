"""Held-out checkpoint selection and early stopping for MARLIN adaptation.

The 100,000-step warm-start run reported a flat ``Exact@1 = 0.0312`` from step
30,000 onward while a matched offline re-evaluation of the same checkpoints
showed the model degrading: candidate return peaked at 0.3438 at step 20,000 and
fell to 0.1562 by step 50,000, with uniqueness (0.3333 -> 0.1354) and mass
validity (0.2419 -> 0.0987) moving with it. All numbers are from
``docs/MARLIN_STATUS_REPORT.md``.

Selection therefore refuses ``Exact@k``. One molecule in 32 was that panel's
resolution floor, and even on the 396-spectrum validation panel the floor is
0.25%, which is smaller than the quantity a selector has to compare. The
admitted metrics are the three that moved together across the degradation, plus
completed validity, which is the decoder-only half of the same signal.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


# Every admitted metric is a rate in [0, 1] that improves upward.
SELECTION_METRICS = (
    "candidate_return_rate",
    "uniqueness",
    "mass_validity",
    "completed_validity",
)

REFUSED_SELECTION_METRICS = {
    "exact_top1": (
        "Exact@1 was flat at 0.0312 from step 20,000 to 70,000 while candidate "
        "return halved; its resolution is 1/32 on the micro panel and 1/396 on "
        "the validation panel"
    ),
    "exact_top10": (
        "Exact@10 equalled Exact@1 in every micro-panel evaluation, so it "
        "carries the same resolution floor"
    ),
}


@dataclass(frozen=True)
class SelectionDecision:
    """The outcome of one periodic evaluation, from the selector's view."""

    step: int
    metric: str
    value: float
    best_step: int
    best_value: float
    improved: bool
    evaluations_since_best: int
    should_stop: bool
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class CheckpointSelector:
    """Track the best checkpoint on a held-out metric and stop when it stalls.

    ``patience`` counts periodic evaluations without an improvement of more than
    ``min_delta``. ``patience = 0`` records the best checkpoint but never stops,
    which is the historical behaviour of every run in this lineage.
    """

    def __init__(
        self,
        *,
        metric: str = "candidate_return_rate",
        patience: int = 0,
        min_delta: float = 0.0,
    ) -> None:
        if metric in REFUSED_SELECTION_METRICS:
            raise ValueError(
                f"{metric} cannot drive checkpoint selection: "
                f"{REFUSED_SELECTION_METRICS[metric]}"
            )
        if metric not in SELECTION_METRICS:
            raise ValueError(
                f"unknown selection metric {metric!r}; "
                f"expected one of {', '.join(SELECTION_METRICS)}"
            )
        if patience < 0:
            raise ValueError("selection patience must be non-negative")
        if min_delta < 0.0:
            raise ValueError("selection min_delta must be non-negative")
        self.metric = metric
        self.patience = patience
        self.min_delta = float(min_delta)
        self.best_step: int | None = None
        self.best_value: float | None = None
        self.evaluations_since_best = 0
        self.history: list[dict[str, Any]] = []

    def metric_value(self, metrics: dict[str, Any]) -> float:
        if self.metric not in metrics:
            raise KeyError(
                f"periodic evaluation metrics have no {self.metric!r}: "
                f"keys={sorted(metrics)[:12]}"
            )
        value = float(metrics[self.metric])
        if not math.isfinite(value):
            raise ValueError(
                f"{self.metric} is not finite ({value}); an empty panel cannot "
                "select a checkpoint"
            )
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{self.metric} must be a rate in [0, 1], got {value}")
        return value

    def update(self, step: int, metrics: dict[str, Any]) -> SelectionDecision:
        value = self.metric_value(metrics)
        improved = (
            self.best_value is None or value > self.best_value + self.min_delta
        )
        if improved:
            self.best_step = step
            self.best_value = value
            self.evaluations_since_best = 0
        else:
            self.evaluations_since_best += 1
        should_stop = self.patience > 0 and self.evaluations_since_best >= self.patience
        reason = None
        if should_stop:
            reason = (
                f"{self.metric} has not improved by more than {self.min_delta} "
                f"in {self.evaluations_since_best} evaluations; best "
                f"{self.best_value} at step {self.best_step}"
            )
        decision = SelectionDecision(
            step=step,
            metric=self.metric,
            value=value,
            best_step=int(self.best_step),
            best_value=float(self.best_value),
            improved=improved,
            evaluations_since_best=self.evaluations_since_best,
            should_stop=should_stop,
            reason=reason,
        )
        self.history.append(decision.as_dict())
        return decision

    def state(self) -> dict[str, Any]:
        """A JSON-serializable record of every evaluation seen so far."""
        return {
            "schema_version": 1,
            "metric": self.metric,
            "mode": "max",
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_step": self.best_step,
            "best_value": self.best_value,
            "evaluations_since_best": self.evaluations_since_best,
            "refused_metrics": dict(REFUSED_SELECTION_METRICS),
            "history": list(self.history),
        }
