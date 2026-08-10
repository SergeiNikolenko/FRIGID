import json

import pytest

from marlin.checkpoint_selection import (
    REFUSED_SELECTION_METRICS,
    CheckpointSelector,
)


def test_selection_refuses_exact_metrics():
    """Exact@1 was flat at 0.0312 from step 20,000 to 70,000 while candidate
    return, uniqueness and mass validity all halved, so it cannot select."""
    for metric in REFUSED_SELECTION_METRICS:
        with pytest.raises(ValueError, match="cannot drive checkpoint selection"):
            CheckpointSelector(metric=metric)


def test_selection_rejects_unknown_metric_and_negative_patience():
    with pytest.raises(ValueError, match="unknown selection metric"):
        CheckpointSelector(metric="tanimoto_top1")
    with pytest.raises(ValueError, match="patience must be non-negative"):
        CheckpointSelector(patience=-1)
    with pytest.raises(ValueError, match="min_delta must be non-negative"):
        CheckpointSelector(min_delta=-0.1)


def test_selection_tracks_the_measured_candidate_return_trajectory():
    """The matched offline re-evaluation of the reference run, from
    docs/MARLIN_STATUS_REPORT.md. Patience 3 has to stop after step 50,000 and
    keep step 20,000 as the best checkpoint."""
    trajectory = {
        10000: 0.0000,
        20000: 0.3438,
        30000: 0.2812,
        40000: 0.2188,
        50000: 0.1562,
        60000: 0.3125,
        70000: 0.1875,
    }
    selector = CheckpointSelector(metric="candidate_return_rate", patience=3)
    stopped_at = None
    for step, value in trajectory.items():
        decision = selector.update(step, {"candidate_return_rate": value})
        if decision.should_stop and stopped_at is None:
            stopped_at = step

    assert stopped_at == 50000
    assert selector.best_step == 20000
    assert selector.best_value == pytest.approx(0.3438)


def test_zero_patience_records_the_best_checkpoint_without_stopping():
    selector = CheckpointSelector(patience=0)
    for step, value in ((1000, 0.4), (2000, 0.1), (3000, 0.1)):
        decision = selector.update(step, {"candidate_return_rate": value})
        assert not decision.should_stop

    assert selector.best_step == 1000
    assert selector.evaluations_since_best == 2


def test_min_delta_ignores_movement_inside_the_panel_noise_band():
    selector = CheckpointSelector(patience=2, min_delta=0.05)
    selector.update(1000, {"candidate_return_rate": 0.30})
    inside = selector.update(2000, {"candidate_return_rate": 0.33})
    stop = selector.update(3000, {"candidate_return_rate": 0.34})

    assert not inside.improved
    assert stop.should_stop
    assert selector.best_step == 1000


def test_selection_rejects_a_missing_or_unusable_metric():
    selector = CheckpointSelector(metric="uniqueness")
    with pytest.raises(KeyError):
        selector.update(1000, {"candidate_return_rate": 0.1})
    with pytest.raises(ValueError, match="not finite"):
        selector.update(1000, {"uniqueness": float("nan")})
    with pytest.raises(ValueError, match="rate in "):
        selector.update(1000, {"uniqueness": 1.5})


def test_selection_state_is_serializable_history():
    selector = CheckpointSelector(metric="mass_validity", patience=1)
    selector.update(1000, {"mass_validity": 0.2})
    selector.update(2000, {"mass_validity": 0.1})

    state = json.loads(json.dumps(selector.state()))

    assert state["metric"] == "mass_validity"
    assert state["best_step"] == 1000
    assert [entry["step"] for entry in state["history"]] == [1000, 2000]
    assert state["history"][-1]["should_stop"] is True
    assert "exact_top1" in state["refused_metrics"]
