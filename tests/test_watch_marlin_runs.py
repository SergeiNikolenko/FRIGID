"""Tests for the run dashboard.

Everything the dashboard decides - which kind of run this is, where it has got
to, and whether it is healthy - is a pure function of a snapshot, so none of
these tests need a ClearML server. The ClearML boundary itself is exercised
only through `parse_worker_stats`, which is the one place a server response is
reshaped.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import watch_marlin_runs as watch  # noqa: E402


NOW = datetime(2026, 8, 12, 20, 0, 0, tzinfo=timezone.utc)

EVAL_CONTAINER = (
    "-v /mnt/netstorage:/mnt/netstorage --shm-size 32g "
    "-e MARLIN_PAYLOAD_TASK_ID=3993339892da42c9be34c7e76022376e "
    "-e MARLIN_RUN_NAME=clean-new-oracle-c8 -e MARLIN_CANDIDATES=8 "
    "-e MARLIN_CAP=300 -e MARLIN_SHARDS=6 -e MARLIN_FP_MODE=oracle "
    "-e PYTHONFAULTHANDLER=1 -e CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL=1"
)
TRAIN_CONTAINER = (
    "-v /mnt/netstorage:/mnt/netstorage --shm-size 32g "
    "-e MARLIN_SOFT_FINGERPRINT=1 -e MARLIN_MAX_STEPS=20000 "
    "-e MARLIN_EVALUATION_MANIFEST=configs/benchmarks/nplib1_v1/nplib1_val_clean322_v1.tsv "
    "-e MARLIN_EVALUATION_SPECTRA=321 -e PYTHONFAULTHANDLER=1 "
)
HEARTBEATS = "\n".join(
    f"[heartbeat 2026-08-12T{h:02d}:{m:02d}:43Z] shards_finished=0/6 rows={rows} elapsed={elapsed}s"
    for h, m, rows, elapsed in [
        (17, 51, 5, 612),
        (18, 1, 12, 1212),
        (18, 11, 19, 1812),
        (18, 21, 31, 2412),
    ]
)


def make_snapshot(**overrides) -> watch.RunSnapshot:
    defaults = dict(
        task_id="t" * 32,
        name="marlin-run",
        status="in_progress",
        kind=watch.EVALUATION,
        queue_id="q1",
        queue_name="sience",
        worker="aiagent03:gpu0",
        started=NOW - timedelta(hours=2),
        last_update=NOW - timedelta(minutes=1),
        active_duration=7200.0,
    )
    defaults.update(overrides)
    return watch.RunSnapshot(**defaults)


def worker(**overrides) -> watch.WorkerInfo:
    defaults = dict(
        worker_id="aiagent03:gpu0",
        queues=["high_q", "sience"],
        task_id="t" * 32,
        last_activity=NOW - timedelta(seconds=30),
        disk_free_percent=45.0,
    )
    defaults.update(overrides)
    return watch.WorkerInfo(**defaults)


# -- parsing ---------------------------------------------------------------


def test_container_env_is_parsed_from_docker_arguments():
    env = watch.parse_container_env(EVAL_CONTAINER)
    assert env["MARLIN_RUN_NAME"] == "clean-new-oracle-c8"
    assert env["MARLIN_SHARDS"] == "6"
    assert env["MARLIN_FP_MODE"] == "oracle"
    # The volume mount is not an -e pair and must not be mistaken for one.
    assert "-v" not in env


def test_container_env_of_a_task_without_a_container_is_empty():
    assert watch.parse_container_env(None) == {}
    assert watch.parse_container_env("") == {}


@pytest.mark.parametrize(
    ("entry_point", "container", "name", "expected"),
    [
        ("marlin_clean_panel_eval.sh", EVAL_CONTAINER, "marlin-B", watch.EVALUATION),
        (
            "scripts/run_marlin_faro_spectrum_adaptation.sh",
            TRAIN_CONTAINER,
            "marlin-T1a",
            watch.TRAINING,
        ),
        ("", TRAIN_CONTAINER, "anonymous", watch.TRAINING),
        ("", EVAL_CONTAINER, "anonymous", watch.EVALUATION),
        ("", "", "marlin-eval-payload-w3", watch.EVALUATION),
        ("", "", "something-else", watch.UNKNOWN),
    ],
)
def test_run_kind_is_decided_on_what_the_task_was_told_to_do(
    entry_point, container, name, expected
):
    env = watch.parse_container_env(container)
    assert watch.classify_run(entry_point, env, name) == expected


def test_heartbeats_are_read_in_order_with_rows_and_elapsed():
    beats = watch.parse_heartbeats(HEARTBEATS)
    assert [beat.rows for beat in beats] == [5, 12, 19, 31]
    assert beats[-1].elapsed_seconds == 2412
    assert beats[-1].shards_total == 6
    assert beats[0].at == datetime(2026, 8, 12, 17, 51, 43, tzinfo=timezone.utc)


def test_console_without_heartbeats_yields_none():
    assert watch.parse_heartbeats("Process completed successfully") == []


def test_panel_path_comes_from_the_task_script():
    diff = (
        'python "$CODE/scripts/evaluate_marlin_nplib1.py" '
        '--spec-manifest "$PANEL" '
        "$CODE/configs/benchmarks/nplib1_v1/nplib1_val_clean322_v1.tsv"
    )
    assert watch.parse_panel_path(diff) == (
        "configs/benchmarks/nplib1_v1/nplib1_val_clean322_v1.tsv"
    )
    assert watch.parse_panel_path("") is None


def test_panel_spectra_count_reads_the_manifest_in_this_checkout(tmp_path):
    manifest = tmp_path / "configs" / "benchmarks" / "p.tsv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("spec_name\ta\nA\t1\nB\t2\n\n")
    assert watch.panel_spectra_count("configs/benchmarks/p.tsv", tmp_path) == 2
    # A denominator we cannot resolve stays absent rather than being invented.
    assert watch.panel_spectra_count("configs/benchmarks/missing.tsv", tmp_path) is None


def test_the_real_clean_panel_manifest_has_321_spectra():
    """The panel the two live evaluations run on, per `docs/DECODER_PROGRAM.md` §2."""
    assert (
        watch.panel_spectra_count(
            "configs/benchmarks/nplib1_v1/nplib1_val_clean322_v1.tsv"
        )
        == 321
    )


def test_worker_stats_reduce_to_the_last_value_of_each_metric():
    payload = {
        "workers": [
            {
                "worker": "aiagent03:gpu0",
                "metrics": [
                    {"metric": "disk_free_home", "stats": [{"values": [12.0, 11.4]}]},
                    {"metric": "gpu_usage", "stats": [{"values": [1.7, 2.5]}]},
                ],
            }
        ]
    }
    assert watch.parse_worker_stats(payload) == {
        "aiagent03:gpu0": {"disk_free_home": 11.4, "gpu_usage": 2.5}
    }


def test_worker_stats_survive_an_empty_response():
    assert watch.parse_worker_stats({}) == {}
    assert watch.parse_worker_stats({"workers": [{"worker": "w", "metrics": []}]}) == {}


# -- scalars ---------------------------------------------------------------


def test_scalar_summary_reports_last_value_and_direction():
    series = watch.Series("Training metrics", "train_loss", [1, 2, 3], [0.5, 0.4, 0.3])
    summary = watch.summarise_series(series, window=3)
    assert summary["last"] == pytest.approx(0.3)
    assert summary["last_step"] == 3
    assert summary["delta"] == pytest.approx(-0.2)
    assert summary["direction"] == "down"


def test_scalar_summary_of_an_empty_series_has_no_last_value():
    summary = watch.summarise_series(watch.Series("t", "s", [], []))
    assert summary["count"] == 0
    assert summary["last"] is None


def test_series_ranking_puts_loss_first_and_host_telemetry_last():
    scalars = [
        watch.Series(":monitor:gpu", "gpu_0_utilization", [1], [3.0]),
        watch.Series("Training metrics", "learning_rate", [1], [1e-5]),
        watch.Series("Training metrics", "train_loss", [1], [0.4]),
    ]
    assert [s.name for s in watch.rank_series(scalars)] == [
        "train_loss",
        "learning_rate",
        "gpu_0_utilization",
    ]


# -- progress --------------------------------------------------------------


def test_training_progress_counts_steps_against_the_step_budget():
    snapshot = make_snapshot(
        kind=watch.TRAINING,
        env=watch.parse_container_env(TRAIN_CONTAINER),
        scalars=[
            watch.Series("Training metrics", "train_loss", [500, 1000], [0.5, 0.4])
        ],
        active_duration=3600.0,
    )
    progress = watch.training_progress(snapshot)
    assert progress.unit == "steps"
    assert progress.current == 1000
    assert progress.total == 20000
    assert progress.fraction == pytest.approx(0.05)
    assert progress.rate_per_hour == pytest.approx(1000.0)
    assert progress.eta_seconds == pytest.approx(19 * 3600.0)


def test_training_progress_ignores_clearml_host_telemetry_iterations():
    snapshot = make_snapshot(
        kind=watch.TRAINING,
        env=watch.parse_container_env(TRAIN_CONTAINER),
        scalars=[
            watch.Series("Training metrics", "train_loss", [500], [0.5]),
            watch.Series(":monitor:gpu", "gpu_0_utilization", [999999], [3.0]),
        ],
    )
    assert watch.training_progress(snapshot).current == 500


def test_evaluation_progress_falls_back_to_the_heartbeat_when_no_file_is_reachable():
    snapshot = make_snapshot(
        env=watch.parse_container_env(EVAL_CONTAINER),
        console=[HEARTBEATS],
        heartbeats=watch.parse_heartbeats(HEARTBEATS),
        panel_spectra=321,
    )
    progress = watch.evaluation_progress(snapshot)
    assert progress.unit == "rows"
    assert progress.current == 31
    assert progress.total == 321
    assert progress.source == "console heartbeat"
    # 12 -> 31 rows over 1212 -> 2412 s, first beat dropped as startup.
    assert progress.rate_per_hour == pytest.approx(57.0)


def test_evaluation_progress_prefers_the_predictions_file_and_names_the_disagreement():
    snapshot = make_snapshot(
        env=watch.parse_container_env(EVAL_CONTAINER),
        console=[HEARTBEATS],
        heartbeats=watch.parse_heartbeats(HEARTBEATS),
        panel_spectra=321,
        partial_metrics={"total_spectra": 28},
    )
    progress = watch.evaluation_progress(snapshot)
    assert progress.current == 28
    assert progress.source == "predictions file"
    assert progress.secondary == ("console heartbeat", 31.0)
    assert "console heartbeat says 31" in watch.render_progress(progress)


def test_a_smoke_run_is_scored_against_its_truncated_target_not_the_panel():
    """`MARLIN_MAX_SPECTRA` truncates each shard, so the denominator is not 321."""
    env = watch.parse_container_env(EVAL_CONTAINER + " -e MARLIN_MAX_SPECTRA=2")
    assert watch.evaluation_target_rows(321, env) == 12
    assert (
        watch.evaluation_target_rows(321, watch.parse_container_env(EVAL_CONTAINER))
        == 321
    )
    # The truncation can never exceed the panel it was cut from.
    big = watch.parse_container_env(EVAL_CONTAINER + " -e MARLIN_MAX_SPECTRA=500")
    assert watch.evaluation_target_rows(321, big) == 321


def test_progress_of_a_run_with_no_signal_is_unknown_not_zero():
    progress = watch.run_progress(make_snapshot(kind=watch.UNKNOWN))
    assert progress.current is None
    assert "unknown" in watch.render_progress(progress)


# -- partial predictions ---------------------------------------------------


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def row(spec: str, *, exact: bool, returned: bool, tanimoto: float = 0.0) -> dict:
    return {
        "spec_name": spec,
        "exact_top1": exact,
        "exact_top10": exact,
        "candidate_returned": returned,
        "tanimoto_top1": tanimoto,
        "tanimoto_top10": tanimoto,
        "formula_top1": returned,
        "attempts": 8,
        "truncated": False,
        "candidates": [],
    }


def test_partial_metrics_are_scored_from_the_shards_before_any_merge(tmp_path):
    root = tmp_path / "evaluations"
    write_rows(
        root / "run-a" / "shard00" / "predictions.jsonl",
        [
            row("A", exact=True, returned=True, tanimoto=0.9),
            row("B", exact=False, returned=False),
        ],
    )
    write_rows(
        root / "run-a" / "shard01" / "predictions.jsonl",
        [row("C", exact=False, returned=True, tanimoto=0.3)],
    )
    metrics, path, note = watch.partial_metrics_for("run-a", [root])
    assert note == ""
    assert path.name == "predictions.jsonl"
    assert metrics["total_spectra"] == 3
    assert metrics["exact_match_top1"] == pytest.approx(1 / 3)
    assert metrics["candidate_return_rate"] == pytest.approx(2 / 3)


def test_a_spectrum_present_in_both_the_merge_and_a_shard_is_counted_once(tmp_path):
    root = tmp_path / "evaluations"
    write_rows(
        root / "run-a" / "predictions.jsonl", [row("A", exact=True, returned=True)]
    )
    write_rows(
        root / "run-a" / "shard00" / "predictions.jsonl",
        [row("A", exact=True, returned=True)],
    )
    metrics, _, _ = watch.partial_metrics_for("run-a", [root])
    assert metrics["total_spectra"] == 1


def test_a_half_written_final_line_does_not_break_the_read(tmp_path):
    """The file is being appended to while it is read, so truncation is normal."""
    root = tmp_path / "evaluations"
    path = root / "run-a" / "predictions.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(row("A", exact=True, returned=True)) + '\n{"spec_name": "B"'
    )
    metrics, _, _ = watch.partial_metrics_for("run-a", [root])
    assert metrics["total_spectra"] == 1


def test_an_unreachable_predictions_file_says_so_rather_than_reporting_zero(tmp_path):
    metrics, path, note = watch.partial_metrics_for("run-a", [tmp_path])
    assert metrics is None
    assert path is None
    assert "not reachable" in note


def test_a_task_without_a_run_name_is_reported_as_such():
    metrics, _, note = watch.partial_metrics_for(None, [Path("/nowhere")])
    assert metrics is None
    assert "no run name" in note


# -- health ----------------------------------------------------------------


def health(
    snapshot, progress=None, workers=None, queue_counts=None, previous=None, **kw
):
    return watch.check_health(
        snapshot,
        progress or watch.run_progress(snapshot),
        workers if workers is not None else {"aiagent03:gpu0": worker()},
        queue_counts if queue_counts is not None else {"q1": 2},
        NOW,
        previous,
        **kw,
    )


def codes(flags) -> set[str]:
    return {flag.code for flag in flags}


def test_a_queue_with_no_worker_is_critical():
    snapshot = make_snapshot(status="queued", worker=None, queue_position=1)
    flags = health(snapshot, queue_counts={"q1": 0})
    assert codes(flags) == {"queue-has-no-worker"}
    assert flags[0].level == watch.CRITICAL


def test_a_queued_run_behind_busy_workers_is_only_informational():
    snapshot = make_snapshot(status="queued", worker=None, queue_position=2)
    flags = health(snapshot, queue_counts={"q1": 2})
    assert codes(flags) == {"queued-behind"}
    assert flags[0].level == watch.INFO
    assert "position 2" in flags[0].message


def test_a_running_task_with_no_worker_recorded_is_critical():
    flags = health(make_snapshot(worker=None), workers={})
    assert "no-worker-attached" in codes(flags)


def test_a_running_task_whose_worker_went_silent_is_critical():
    silent = worker(last_activity=NOW - timedelta(minutes=45))
    flags = health(make_snapshot(), workers={"aiagent03:gpu0": silent})
    attached = [f for f in flags if f.code == "no-worker-attached"]
    assert attached and attached[0].level == watch.CRITICAL


def test_rows_that_have_not_moved_for_thirty_minutes_are_stalled():
    stuck = "\n".join(
        [
            "[heartbeat 2026-08-12T18:00:00Z] shards_finished=0/6 rows=40 elapsed=1200s",
            "[heartbeat 2026-08-12T18:10:00Z] shards_finished=0/6 rows=73 elapsed=1800s",
            "[heartbeat 2026-08-12T18:20:00Z] shards_finished=0/6 rows=73 elapsed=2400s",
            "[heartbeat 2026-08-12T18:30:00Z] shards_finished=0/6 rows=73 elapsed=3000s",
            "[heartbeat 2026-08-12T18:40:00Z] shards_finished=0/6 rows=73 elapsed=3600s",
            "[heartbeat 2026-08-12T18:50:00Z] shards_finished=0/6 rows=73 elapsed=4200s",
        ]
    )
    snapshot = make_snapshot(
        env=watch.parse_container_env(EVAL_CONTAINER),
        console=[stuck],
        heartbeats=watch.parse_heartbeats(stuck),
        panel_spectra=321,
    )
    flags = health(snapshot)
    stalled = [f for f in flags if f.code == "progress-stalled"]
    assert stalled and stalled[0].level == watch.CRITICAL
    assert "73" in stalled[0].message


def test_rows_still_advancing_are_not_stalled():
    snapshot = make_snapshot(
        env=watch.parse_container_env(EVAL_CONTAINER),
        console=[HEARTBEATS],
        heartbeats=watch.parse_heartbeats(HEARTBEATS),
        panel_spectra=321,
    )
    assert "progress-stalled" not in codes(health(snapshot))


def test_a_training_step_count_frozen_since_the_last_invocation_is_stalled():
    """Training arms carry no timestamped counter, so the state file supplies one."""
    snapshot = make_snapshot(
        kind=watch.TRAINING,
        env=watch.parse_container_env(TRAIN_CONTAINER),
        scalars=[watch.Series("Training metrics", "train_loss", [4000], [0.4])],
    )
    previous = {
        "unit": "steps",
        "current": 4000,
        "first_seen_at": (NOW - timedelta(minutes=95)).isoformat(),
    }
    flags = health(snapshot, previous=previous)
    stalled = [f for f in flags if f.code == "progress-stalled"]
    assert stalled and "4000" in stalled[0].message


def test_state_keeps_the_first_sighting_of_an_unchanged_counter(tmp_path):
    progress = watch.Progress(unit="steps", current=4000, total=20000, source="s")
    state = watch.update_state({}, "task", progress, NOW - timedelta(minutes=40))
    state = watch.update_state(state, "task", progress, NOW)
    assert state["task"]["first_seen_at"] == (NOW - timedelta(minutes=40)).isoformat()

    advanced = watch.Progress(unit="steps", current=4500, total=20000, source="s")
    state = watch.update_state(state, "task", advanced, NOW)
    assert state["task"]["first_seen_at"] == NOW.isoformat()

    path = tmp_path / "state.json"
    watch.save_state(path, state)
    assert watch.load_state(path)["task"]["current"] == 4500


def test_load_state_tolerates_a_missing_or_corrupt_file(tmp_path):
    assert watch.load_state(tmp_path / "absent.json") == {}
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json")
    assert watch.load_state(corrupt) == {}


def test_a_nan_training_loss_is_critical():
    scalars = [
        watch.Series("Training metrics", "train_loss", [100, 200], [0.5, float("nan")])
    ]
    flags = watch.check_loss(scalars)
    assert [f.code for f in flags] == ["loss-nan"]
    assert flags[0].level == watch.CRITICAL


def test_a_flat_training_loss_is_a_warning():
    scalars = [
        watch.Series("Training metrics", "train_loss", list(range(5)), [0.4] * 5)
    ]
    flags = watch.check_loss(scalars)
    assert [f.code for f in flags] == ["loss-flat"]
    assert flags[0].level == watch.WARNING


def test_a_moving_training_loss_raises_nothing():
    scalars = [
        watch.Series(
            "Training metrics",
            "train_loss",
            list(range(5)),
            [0.5, 0.48, 0.45, 0.43, 0.4],
        )
    ]
    assert watch.check_loss(scalars) == []


def test_loss_checks_ignore_series_that_are_not_losses():
    scalars = [
        watch.Series("Training metrics", "learning_rate", list(range(5)), [1e-5] * 5)
    ]
    assert watch.check_loss(scalars) == []


def test_a_nearly_full_worker_disk_is_flagged_at_two_levels():
    warn = watch.check_worker_disk({"w": worker(disk_free_percent=11.4)})
    assert [f.level for f in warn] == [watch.WARNING]
    assert "11.4% disk free" in warn[0].message

    crit = watch.check_worker_disk({"w": worker(disk_free_percent=3.0)})
    assert [f.level for f in crit] == [watch.CRITICAL]

    assert watch.check_worker_disk({"w": worker(disk_free_percent=45.0)}) == []
    # A worker that reports no disk metric must not be flagged as full.
    assert watch.check_worker_disk({"w": worker(disk_free_percent=None)}) == []


def test_a_run_only_carries_the_disk_of_the_worker_it_is_on():
    workers = {
        "aiagent03:gpu0": worker(disk_free_percent=11.4),
        "aiagent01:gpu0": worker(worker_id="aiagent01:gpu0", disk_free_percent=2.0),
    }
    flags = health(make_snapshot(), workers=workers)
    disk = [f for f in flags if f.code == "worker-disk-low"]
    assert len(disk) == 1
    assert "aiagent03:gpu0" in disk[0].message


def test_a_finished_run_raises_no_liveness_flags():
    snapshot = make_snapshot(status="completed", last_update=NOW - timedelta(days=3))
    assert codes(health(snapshot)) == set()


def test_flags_are_ordered_worst_first():
    snapshot = make_snapshot(worker=None)
    flags = health(
        snapshot,
        workers={"aiagent03:gpu0": worker(disk_free_percent=11.4)},
    )
    levels = [watch._LEVEL_ORDER[f.level] for f in flags]
    assert levels == sorted(levels)


# -- selection and rendering ----------------------------------------------


class FakeTask:
    def __init__(self, task_id, status, last_update):
        self.id = task_id
        self.status = status
        self.last_update = last_update


def test_selection_defaults_to_active_runs_plus_recent_history():
    tasks = [
        FakeTask("a", "in_progress", NOW - timedelta(days=9)),
        FakeTask("b", "queued", NOW - timedelta(days=9)),
        FakeTask("c", "failed", NOW - timedelta(hours=2)),
        FakeTask("d", "completed", NOW - timedelta(days=4)),
    ]
    chosen = {t.id for t in watch.select_tasks(tasks, None, 12.0, NOW)}
    assert chosen == {"a", "b", "c"}


def test_selection_by_explicit_task_id_ignores_age_and_status():
    tasks = [
        FakeTask("a", "completed", NOW - timedelta(days=90)),
        FakeTask("b", "queued", NOW),
    ]
    chosen = watch.select_tasks(tasks, None, 1.0, NOW, task_ids=["a"])
    assert [t.id for t in chosen] == ["a"]


def test_selection_by_status_overrides_the_age_window():
    tasks = [
        FakeTask("a", "failed", NOW - timedelta(days=90)),
        FakeTask("b", "queued", NOW),
    ]
    chosen = watch.select_tasks(tasks, ["failed"], 1.0, NOW)
    assert [t.id for t in chosen] == ["a"]


def test_the_report_states_progress_its_source_and_every_flag():
    snapshot = make_snapshot(
        name="marlin-B-full-clean-panel-ORACLE-c8",
        env=watch.parse_container_env(EVAL_CONTAINER),
        console=[HEARTBEATS],
        heartbeats=watch.parse_heartbeats(HEARTBEATS),
        panel_spectra=321,
        predictions_note="predictions file not reachable from this host",
    )
    progress = watch.run_progress(snapshot)
    flags = health(
        snapshot, progress, workers={"aiagent03:gpu0": worker(disk_free_percent=11.4)}
    )
    text = watch.render_run(
        snapshot,
        progress,
        flags,
        {"aiagent03:gpu0": worker(disk_free_percent=11.4)},
        NOW,
        watch.Palette(False),
        console_lines=3,
        scalar_limit=10,
    )
    assert "marlin-B-full-clean-panel-ORACLE-c8" in text
    assert "31/321 rows" in text
    assert "via console heartbeat" in text
    assert "not reachable" in text
    assert "worker-disk-low" in text
    assert "rows=31" in text  # console tail is really the console


def test_partial_metrics_are_labelled_partial_while_the_run_is_alive():
    snapshot = make_snapshot(
        status="in_progress",
        panel_spectra=321,
        partial_metrics={
            "total_spectra": 118,
            "exact_match_top1": 0.0339,
            "exact_match_top10": 0.0339,
            "candidate_return_rate": 0.4152,
            "tanimoto_top1_mean": 0.1611,
            "truncated_spectra": 4,
        },
    )
    text = watch.render_partial_metrics(snapshot, watch.Palette(False))
    assert text.startswith("PARTIAL over 118 rows/321")
    assert "Exact@1 3.39%" in text
    assert "return rate 41.52%" in text
    assert "truncated 4" in text


def test_a_finished_full_panel_is_labelled_final():
    snapshot = make_snapshot(
        status="completed",
        panel_spectra=321,
        partial_metrics={
            "total_spectra": 321,
            "exact_match_top1": 0.0093,
            "exact_match_top10": 0.0093,
            "candidate_return_rate": 0.4143,
            "tanimoto_top1_mean": 0.1575,
            "truncated_spectra": 0,
        },
    )
    text = watch.render_partial_metrics(snapshot, watch.Palette(False))
    assert text.startswith("FINAL over 321 rows/321")


def test_the_json_payload_carries_progress_metrics_and_health():
    snapshot = make_snapshot(
        env=watch.parse_container_env(EVAL_CONTAINER),
        console=[HEARTBEATS],
        heartbeats=watch.parse_heartbeats(HEARTBEATS),
        panel_spectra=321,
    )
    progress = watch.run_progress(snapshot)
    payload = watch.snapshot_to_dict(snapshot, progress, health(snapshot, progress), 2)
    assert payload["progress"]["current"] == 31
    assert payload["progress"]["source"] == "console heartbeat"
    assert payload["panel"]["spectra"] == 321
    assert len(payload["console_tail"]) == 2
    assert json.loads(json.dumps(payload, default=str))["kind"] == watch.EVALUATION


def test_console_tail_drops_blank_lines_and_keeps_the_last_ones():
    assert watch.console_tail(["a\n\nb", "c\n"], 2) == ["b", "c"]
    assert watch.console_tail([], 5) == []
    assert watch.console_tail(["a"], 0) == []
