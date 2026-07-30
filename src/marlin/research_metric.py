"""Stage-aware molecular research metric for MARLIN.

Exact retrieval remains the final objective.  Before Exact is non-zero, the
metric promotes candidates that cross the prerequisite generation gates rather
than treating every run as an indistinguishable zero.
"""

from __future__ import annotations

import math
from typing import Any


RATE_METRICS = (
    "exact_top1",
    "exact_top10",
    "candidate_return_rate",
    "validity",
    "mass_validity",
    "tanimoto_top10",
)


def _rate(metrics: dict[str, Any], name: str) -> float:
    value = float(metrics.get(name, 0.0))
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite rate in [0, 1], got {value}")
    return value


def staged_research_metric(metrics: dict[str, Any]) -> dict[str, float | str]:
    """Return a lexicographic score for prerequisite molecular gates.

    Score bands are deliberately disjoint:

    - ``[0, 1]``: syntactically valid decoding;
    - ``(1, 2]``: at least one mass-valid decoded candidate;
    - ``(2, 3]``: strict mass-shell candidate return;
    - ``(3, 4]``: non-zero Exact retrieval.

    This score is for validation research selection only.  Exact@1 and
    Exact@10 remain the reported paper metrics.
    """
    values = {name: _rate(metrics, name) for name in RATE_METRICS}
    exact_score = (
        0.6 * values["exact_top1"] + 0.4 * values["exact_top10"]
    )

    if values["exact_top1"] > 0.0 or values["exact_top10"] > 0.0:
        stage = "exact_retrieval"
        score = 3.0 + exact_score
    elif (
        values["candidate_return_rate"] > 0.0
        and values["mass_validity"] > 0.0
    ):
        stage = "mass_shell_return"
        score = 2.0 + (
            0.4 * values["candidate_return_rate"]
            + 0.4 * values["mass_validity"]
            + 0.2 * values["tanimoto_top10"]
        )
    elif values["mass_validity"] > 0.0:
        stage = "mass_valid_decoding"
        score = 1.0 + (
            0.7 * values["mass_validity"] + 0.3 * values["validity"]
        )
    else:
        stage = "valid_decoding"
        score = values["validity"]

    return {
        "research_stage": stage,
        "research_score": score,
        "exact_score": exact_score,
    }
