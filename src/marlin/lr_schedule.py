"""A warmup-then-cosine schedule for MARLIN adaptation, and its derivation.

Every adaptation run in this lineage so far had **no schedule at all**. The
control-r2 checkpoint the queued arms warm-start from carries
``lr_schedulers: []`` and ``global_step = 100000`` at a constant ``1e-5``
(measured from
``/mnt/netstorage/nikolenko/marlin/cache/checkpoints/control-r2/
aed408c7d2c01c86a4b257e5119c28b11404971c3df3fd09a76c054ef0e7b14f/
step=100000.ckpt``).

Its own ancestor, the released DLM checkpoint
(``/mnt/netstorage/nikolenko/marlin/checkpoints/frigid/DLM.ckpt``,
sha256 ``b6177c2d43448380aba80ff41c01461ea34ca2ca93b213986954c5afb7f0f457``),
stores the schedule that produced its weights::

    {"_milestones": [6000], "last_epoch": 520000,
     "_last_lr": [5.2697058404552555e-08],
     "_schedulers": [
        {"start_factor": 1e-06, "end_factor": 1.0, "total_iters": 6000,
         "base_lrs": [0.00013], ...},
        {"T_max": 520000, "eta_min": 1e-08, "base_lrs": [0.00013],
         "last_epoch": 514000, ...}]}

So the released weights were left at ``5.2697058404552555e-08``, and we resume
them at ``1e-5`` -- ``190x`` -- flat forever.

Deriving the peak
-----------------

AdamW's per-step update is ``lr * m_hat / (sqrt(v_hat) + eps)``, whose magnitude
is of order ``lr`` whenever the gradient sign is consistent. The worst-case
cumulative displacement of any single weight over a run is therefore bounded by
the **sum of the schedule**, ``sum_t lr_t``. That sum is the quantity to budget,
and the weight scale it should be compared against is measurable: the RMS of the
173,801,752 float parameters under ``decoder.`` in the control-r2 checkpoint is
``0.082472829079876``.

Three anchors, all arithmetic on the schedule above:

1. **Pure continuation.** Our 20,000 steps at global batch 256 are 5.12M
   examples, which the released run covered in 2,667 of its steps at batch
   1,920. Replaying its schedule over that stretch runs from
   ``5.2697e-8`` down to ``1.8438e-8``. Nothing below that changes anything.
2. **A full retrain at our batch.** The largest rate these weights have ever
   seen is ``1.3e-4`` at batch 1,920; the linear scaling rule puts that at
   ``1.3e-4 * 256 / 1920 = 1.7333e-5`` for our batch. **Today's constant
   ``1e-5`` is 58% of that** -- the current "fine-tune" runs at a pretraining
   peak, without the anneal that made it safe.
3. **The released run's final decade** -- the stretch over which its own
   learning rate fell from ``10x`` the terminal value to the terminal value.
   That is the last well-defined segment of the released schedule in which the
   optimiser was still making updates of the order of magnitude we are about to
   make, and its displacement budget is the one this module adopts.

``derive_peak_learning_rate`` implements anchor 3 exactly: it sums the released
schedule over that segment and solves for the peak whose warmup-then-cosine over
our own step budget has the same sum. At 20,000 steps, 1,000 warmup steps and a
floor equal to the checkpoint's terminal rate that peak is ``3.2e-7`` --
``6.1x`` the terminal rate rather than ``190x``, and ``54x`` below the
batch-scaled pretraining peak.

Deriving the warmup
-------------------

The optimizer state is **not** restored from the checkpoint, so AdamW starts
with zero moments. Its second-moment estimate has an effective averaging window
of ``1 / (1 - beta2) = 1000`` steps at the default ``beta2 = 0.999``; before
that window has filled, ``sqrt(v_hat)`` is dominated by whichever few batches
happened to come first and the update is not the update the peak was chosen
for. 1,000 steps is therefore the shortest warmup that is not a guess. The
released run used 6,000, or 1.15% of its length; ours is 5% of a 20,000-step
run.
"""

from __future__ import annotations

import math

import torch


# Read verbatim from DLM.ckpt["lr_schedulers"][0]; see the module docstring.
RELEASED_BASE_LEARNING_RATE = 1.3e-4
RELEASED_WARMUP_STEPS = 6000
RELEASED_WARMUP_START_FACTOR = 1e-6
RELEASED_COSINE_T_MAX = 520000
RELEASED_COSINE_ETA_MIN = 1e-8
RELEASED_COSINE_LAST_STEP = 514000
RELEASED_TERMINAL_LEARNING_RATE = 5.2697058404552555e-08
RELEASED_GLOBAL_BATCH_SIZE = 1920

# RMS of the 173,801,752 float parameters under "decoder." in the control-r2
# checkpoint the queued arms warm-start from.
SOURCE_WEIGHT_RMS = 0.082472829079876

# The default warmup: AdamW's second-moment averaging window at beta2 = 0.999.
ADAMW_SECOND_MOMENT_WINDOW = 1000


def released_learning_rate(step: int) -> float:
    """The released schedule's learning rate at one of its optimizer steps.

    ``step`` counts optimizer steps from the start of pretraining: a linear
    warmup for the first ``RELEASED_WARMUP_STEPS``, then a cosine whose own
    coordinate restarts at the milestone, which is why the checkpoint's cosine
    reads ``last_epoch = 514000`` at ``global_step = 520000``.
    """
    if step < 0:
        raise ValueError("released schedule step must be non-negative")
    if step < RELEASED_WARMUP_STEPS:
        factor = RELEASED_WARMUP_START_FACTOR + (
            1.0 - RELEASED_WARMUP_START_FACTOR
        ) * (step / RELEASED_WARMUP_STEPS)
        return RELEASED_BASE_LEARNING_RATE * factor
    cosine_step = min(step - RELEASED_WARMUP_STEPS, RELEASED_COSINE_T_MAX)
    return RELEASED_COSINE_ETA_MIN + (
        RELEASED_BASE_LEARNING_RATE - RELEASED_COSINE_ETA_MIN
    ) * (1 + math.cos(math.pi * cosine_step / RELEASED_COSINE_T_MAX)) / 2


def released_final_decade_steps() -> tuple[int, int]:
    """The released cosine's span from 10x its terminal rate to its terminal rate.

    Returned in the pretraining step coordinate, as a half-open ``[first, last)``
    pair, so that ``last - first`` is the number of optimizer steps in the
    segment.
    """
    target = 10.0 * RELEASED_TERMINAL_LEARNING_RATE
    last = RELEASED_WARMUP_STEPS + RELEASED_COSINE_LAST_STEP
    low, high = RELEASED_WARMUP_STEPS, last
    # The cosine is monotone decreasing here, so bisect for the first step whose
    # rate has already fallen to the target.
    while low < high:
        middle = (low + high) // 2
        if released_learning_rate(middle) <= target:
            high = middle
        else:
            low = middle + 1
    return low, last


def released_final_decade_displacement() -> float:
    """``sum_t lr_t`` over the released run's final decade of learning rate."""
    first, last = released_final_decade_steps()
    return math.fsum(released_learning_rate(step) for step in range(first, last))


def warmup_cosine_factor(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    floor_factor: float,
    start_factor: float = RELEASED_WARMUP_START_FACTOR,
) -> float:
    """The multiplier a warmup-then-cosine schedule applies at ``step``.

    Linear from ``start_factor`` to 1 over ``warmup_steps``, then cosine from 1
    down to ``floor_factor`` over the remainder. Shaped like the released
    schedule -- ``LinearLR`` into ``CosineAnnealingLR`` -- so the two can be
    compared step for step.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must satisfy 0 <= warmup_steps < total_steps")
    if not 0.0 <= floor_factor <= 1.0:
        raise ValueError("floor_factor must be in [0, 1]")
    if not 0.0 < start_factor <= 1.0:
        raise ValueError("start_factor must be in (0, 1]")
    if step < 0:
        raise ValueError("step must be non-negative")
    if step < warmup_steps:
        return start_factor + (1.0 - start_factor) * (step / warmup_steps)
    progress = min((step - warmup_steps) / (total_steps - warmup_steps), 1.0)
    return floor_factor + (1.0 - floor_factor) * (1 + math.cos(math.pi * progress)) / 2


def schedule_displacement(
    peak: float,
    *,
    total_steps: int,
    warmup_steps: int,
    floor: float,
) -> float:
    """``sum_t lr_t`` over a whole warmup-then-cosine run.

    The bound on how far AdamW can move any one weight over the run; see the
    module docstring.
    """
    if peak <= 0:
        raise ValueError("peak learning rate must be positive")
    if floor < 0 or floor > peak:
        raise ValueError("floor must satisfy 0 <= floor <= peak")
    floor_factor = floor / peak
    return peak * math.fsum(
        warmup_cosine_factor(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            floor_factor=floor_factor,
        )
        for step in range(total_steps)
    )


def derive_peak_learning_rate(
    *,
    total_steps: int,
    warmup_steps: int = ADAMW_SECOND_MOMENT_WINDOW,
    floor: float = RELEASED_TERMINAL_LEARNING_RATE,
    displacement: float | None = None,
) -> float:
    """Solve for the peak whose schedule spends a given displacement budget.

    ``displacement`` defaults to the released run's final decade of learning
    rate (anchor 3 in the module docstring). ``sum_t lr_t`` is affine in the
    peak once the floor is fixed, so this is one linear solve, not a search:

        sum = peak * sum(warmup factors) + sum over the cosine of
              floor + (peak - floor) * cosine(t)

    Raises when the floor alone already spends the budget: that means the run is
    too long for its budget, and the honest fix is fewer steps, not a peak below
    the rate the weights were left at.
    """
    if displacement is None:
        displacement = released_final_decade_displacement()
    if displacement <= 0:
        raise ValueError("displacement budget must be positive")
    at_floor = schedule_displacement(
        floor, total_steps=total_steps, warmup_steps=warmup_steps, floor=floor
    )
    if at_floor >= displacement:
        raise ValueError(
            f"a {total_steps}-step run held at the floor {floor:g} already "
            f"spends {at_floor:g} of a {displacement:g} displacement budget; "
            "shorten the run rather than annealing below the rate the released "
            "weights were left at"
        )
    unit = schedule_displacement(
        1.0, total_steps=total_steps, warmup_steps=warmup_steps, floor=floor
    )
    # sum(peak) = at_floor + (peak - floor) * (unit - at_floor / floor * ... )
    # is fiddly to write in closed form, so use the exact affine interpolation
    # through two evaluated points instead.
    slope = (unit - at_floor) / (1.0 - floor)
    return floor + (displacement - at_floor) / slope


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    floor: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Wrap ``optimizer`` in the warmup-then-cosine schedule derived above.

    The peak is each parameter group's own ``lr``, so a caller sets the peak by
    constructing the optimizer with it. ``LambdaLR`` rather than
    ``SequentialLR`` so that the factor at any step is a pure function that a
    test can evaluate without stepping an optimizer.
    """
    floors = [
        floor / group["lr"] if group["lr"] else 0.0
        for group in optimizer.param_groups
    ]
    for index, group in enumerate(optimizer.param_groups):
        if floor > group["lr"]:
            raise ValueError(
                f"schedule floor {floor:g} is above parameter group {index}'s "
                f"peak {group['lr']:g}"
            )
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            (
                lambda step, floor_factor=floor_factor: warmup_cosine_factor(
                    step,
                    total_steps=total_steps,
                    warmup_steps=warmup_steps,
                    floor_factor=floor_factor,
                )
            )
            for floor_factor in floors
        ],
    )
