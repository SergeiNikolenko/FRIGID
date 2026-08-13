#!/usr/bin/env python3
"""One command that says where every MARLIN run is and whether it is healthy.

The project runs two kinds of job on the ClearML queue and they do not share a
progress signal:

* **Training arms** report optimizer steps as ClearML scalars (`train_loss` and
  friends, published by `marlin.clearml_metrics.ClearMLTrainingMetrics`). Their
  budget is `MARLIN_MAX_STEPS`.
* **Panel evaluations** report nothing as a scalar at all. They shard the panel
  across processes and their only in-band progress signal is the ten-minute
  heartbeat their launcher prints to the console::

      [heartbeat 2026-08-12T19:11:43Z] shards_finished=0/6 rows=73 elapsed=5412s

  Their budget is the row count of the panel TSV named on the evaluator command
  line inside the task's own script diff.

So this script reads both, and prints for every run: state, queue, worker,
elapsed, the last value and trend of every logged scalar, the console tail, and
- for evaluations - the row count plus a **partial** Exact@1 and candidate
return rate scored from the predictions file when that file is reachable.

Reachability is not assumed. On the ClearML workers `/mnt/netstorage` is
worker-local storage that merely shares a path name with the Spectrum NFS root
(`AGENTS.md`, "Shared storage"), so a running worker's predictions file is
usually *not* visible from the login host. When it is not, the row count still
comes from the heartbeat and the metrics line says the file is unreachable
rather than inventing a number.

Health checks flag the failure modes this project has already suffered:

``queue-has-no-worker``   a task sits queued on a queue no worker is serving
``no-worker-attached``    a task claims to run with no worker reporting
``progress-stalled``      rows or steps have not advanced in 30 minutes
``no-recent-report``      nothing at all has been reported in 30 minutes
``loss-nan``              a training loss went NaN or infinite
``loss-flat``             a training loss has not moved across its recent window
``worker-disk-low``       a worker's disk is nearly full

Usage::

    scripts/watch_marlin_runs.py                 # active runs, one shot
    scripts/watch_marlin_runs.py --watch 60      # refresh every 60 s
    scripts/watch_marlin_runs.py --json          # machine readable
    scripts/watch_marlin_runs.py --task-id ID    # one run, any age

`*.innopolis.university` is unreachable while `LD_PRELOAD` is set, so this
script re-executes itself without it rather than reporting the server as down.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from marlin.frigid_convention import frigid_convention_metrics  # noqa: E402

DEFAULT_PROJECT = "MARLIN clean-room reproduction"
DEFAULT_EVALUATION_ROOTS = ("/mnt/netstorage/nikolenko/marlin/evaluations",)
DEFAULT_STATE_FILE = Path.home() / ".cache" / "marlin-watch" / "progress.json"
ACTIVE_STATUSES = ("queued", "in_progress")

CRITICAL = "CRITICAL"
WARNING = "WARNING"
INFO = "INFO"
_LEVEL_ORDER = {CRITICAL: 0, WARNING: 1, INFO: 2}

TRAINING = "training"
EVALUATION = "evaluation"
UNKNOWN = "unknown"

HEARTBEAT_RE = re.compile(
    r"\[heartbeat\s+(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\]\s*"
    r"shards_finished=(?P<done>\d+)/(?P<total>\d+)\s+"
    r"rows=(?P<rows>\d+)\s+"
    r"elapsed=(?P<elapsed>\d+)s"
)
MERGED_ROWS_RE = re.compile(r"merged\s+(?P<rows>\d+)\s+prediction rows")
PANEL_RE = re.compile(r"(configs/benchmarks/[\w./-]+\.tsv)")
DOCKER_ENV_RE = re.compile(r"-e\s+([A-Za-z_][A-Za-z0-9_]*)=(\S*)")


# --------------------------------------------------------------------------
# data


@dataclass
class Series:
    """One ClearML scalar series."""

    title: str
    name: str
    x: list[float]
    y: list[float]

    @property
    def label(self) -> str:
        return f"{self.title}/{self.name}" if self.title else self.name


@dataclass
class Heartbeat:
    at: datetime
    shards_finished: int
    shards_total: int
    rows: int
    elapsed_seconds: int


@dataclass
class Progress:
    """Where a run is, in whatever unit that run counts in."""

    unit: str
    current: float | None
    total: float | None
    source: str
    rate_per_hour: float | None = None
    eta_seconds: float | None = None
    # A second, independent reading of the same counter. Evaluations have two -
    # the predictions file and the console heartbeat - and they can disagree
    # because shards flush at different times. Printing both is the only honest
    # option; silently preferring one hides a stalled shard.
    secondary: tuple[str, float] | None = None

    @property
    def fraction(self) -> float | None:
        if self.current is None or not self.total:
            return None
        return self.current / self.total


@dataclass
class Flag:
    level: str
    code: str
    message: str


@dataclass
class WorkerInfo:
    worker_id: str
    queues: list[str] = field(default_factory=list)
    task_id: str | None = None
    last_activity: datetime | None = None
    disk_free_percent: float | None = None
    gpu_usage_percent: float | None = None


@dataclass
class RunSnapshot:
    """Everything read about one task, before any judgement is applied."""

    task_id: str
    name: str
    status: str
    status_message: str = ""
    kind: str = UNKNOWN
    queue_id: str | None = None
    queue_name: str | None = None
    queue_position: int | None = None
    entry_point: str = ""
    env: dict[str, str] = field(default_factory=dict)
    worker: str | None = None
    started: datetime | None = None
    last_update: datetime | None = None
    active_duration: float | None = None
    scalars: list[Series] = field(default_factory=list)
    console: list[str] = field(default_factory=list)
    heartbeats: list[Heartbeat] = field(default_factory=list)
    panel_path: str | None = None
    panel_spectra: int | None = None
    predictions_path: Path | None = None
    predictions_mtime: datetime | None = None
    partial_metrics: dict[str, Any] | None = None
    predictions_note: str = ""
    url: str | None = None


# --------------------------------------------------------------------------
# pure parsing / summarising


def parse_container_env(arguments: str | None) -> dict[str, str]:
    """Pull ``-e KEY=VALUE`` pairs out of a ClearML container argument string.

    Every parameter of both run kinds is passed this way, so this is where the
    step budget, the panel, the run name and the fingerprint lane live.
    """
    if not arguments:
        return {}
    return {key: value for key, value in DOCKER_ENV_RE.findall(arguments)}


def classify_run(entry_point: str, env: dict[str, str], name: str = "") -> str:
    """Training arm, panel evaluation, or neither.

    Decided on what the task was actually told to do, not on its name: the
    entry point first, then the environment variables only one kind carries.
    """
    entry = (entry_point or "").lower()
    if "eval" in entry:
        return EVALUATION
    if "train" in entry or "adaptation" in entry:
        return TRAINING
    if "MARLIN_MAX_STEPS" in env:
        return TRAINING
    if "MARLIN_FP_MODE" in env or "MARLIN_CANDIDATES" in env:
        return EVALUATION
    lowered = (name or "").lower()
    if "eval" in lowered or "panel" in lowered:
        return EVALUATION
    if "train" in lowered or "adaptation" in lowered:
        return TRAINING
    return UNKNOWN


def parse_heartbeats(console_text: str) -> list[Heartbeat]:
    """Read the evaluation launcher's ten-minute heartbeats in order."""
    beats: list[Heartbeat] = []
    for match in HEARTBEAT_RE.finditer(console_text or ""):
        beats.append(
            Heartbeat(
                at=datetime.strptime(match["ts"], "%Y-%m-%dT%H:%M:%SZ").replace(
                    tzinfo=timezone.utc
                ),
                shards_finished=int(match["done"]),
                shards_total=int(match["total"]),
                rows=int(match["rows"]),
                elapsed_seconds=int(match["elapsed"]),
            )
        )
    beats.sort(key=lambda beat: beat.at)
    return beats


def parse_panel_path(script_diff: str | None) -> str | None:
    """The panel TSV an evaluation was pointed at, from its own script."""
    if not script_diff:
        return None
    matches = PANEL_RE.findall(script_diff)
    return matches[-1] if matches else None


def panel_spectra_count(
    panel_path: str | None, repo_root: Path = REPO_ROOT
) -> int | None:
    """Row count of a panel manifest resolved inside this checkout, or None.

    The manifest is a header plus one row per spectrum. Returning None when the
    file is absent keeps a missing denominator visible instead of guessing one.
    """
    if not panel_path:
        return None
    candidate = repo_root / panel_path
    if not candidate.is_file():
        return None
    rows = [line for line in candidate.read_text().splitlines()[1:] if line.strip()]
    return len(rows)


def summarise_series(series: Series, window: int = 5) -> dict[str, Any]:
    """Last value of a scalar plus how it moved over its recent window."""
    finite = [(x, y) for x, y in zip(series.x, series.y) if y is not None]
    if not finite:
        return {"label": series.label, "count": 0, "last": None}
    last_x, last_y = finite[-1]
    tail = finite[-window:]
    first_y = tail[0][1]
    delta: float | None = None
    direction = "flat"
    if len(tail) > 1 and _is_finite(first_y) and _is_finite(last_y):
        delta = last_y - first_y
        scale = max(abs(first_y), abs(last_y), 1e-12)
        if abs(delta) <= 1e-9 * scale:
            direction = "flat"
        else:
            direction = "down" if delta < 0 else "up"
    return {
        "label": series.label,
        "title": series.title,
        "name": series.name,
        "count": len(finite),
        "last": last_y,
        "last_step": last_x,
        "window": len(tail),
        "delta": delta,
        "direction": direction,
        "finite": _is_finite(last_y),
    }


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def is_machine_monitor(series: Series) -> bool:
    """ClearML's own host telemetry, which is not this project's progress."""
    return series.title.startswith(":monitor:")


def rank_series(scalars: Sequence[Series]) -> list[Series]:
    """Loss first, then the run's own metrics, then ClearML host telemetry.

    A dashboard that truncates its scalar list must truncate the noise, not the
    number the run is judged on.
    """

    def key(series: Series) -> tuple[int, str]:
        if is_machine_monitor(series):
            return (3, series.label)
        if "loss" in series.name:
            return (0, series.label)
        if series.name.startswith("train_") or series.title.startswith("Training"):
            return (1, series.label)
        return (2, series.label)

    return sorted(scalars, key=key)


def training_progress(snapshot: RunSnapshot) -> Progress:
    """Optimizer steps out of ``MARLIN_MAX_STEPS``."""
    total = _as_float(snapshot.env.get("MARLIN_MAX_STEPS"))
    current: float | None = None
    own = [s for s in snapshot.scalars if not is_machine_monitor(s)] or list(
        snapshot.scalars
    )
    for series in own:
        if series.x:
            current = max(current or 0.0, float(series.x[-1]))
    rate = None
    eta = None
    if current and snapshot.active_duration:
        rate = current / (snapshot.active_duration / 3600.0)
        if total and rate > 0:
            eta = (total - current) / rate * 3600.0
    return Progress(
        unit="steps",
        current=current,
        total=total,
        source="clearml scalars",
        rate_per_hour=rate,
        eta_seconds=eta,
    )


def evaluation_target_rows(
    panel_spectra: int | None, env: dict[str, str]
) -> float | None:
    """How many rows this run will decode, which is not always the panel size.

    ``MARLIN_MAX_SPECTRA`` truncates *each shard*, so a smoke run's denominator
    is shards x cap, not the panel. Scoring a smoke against 321 would report a
    finished run as 4% done - the same class of mistake as quoting a truncated
    evaluation as a panel result (`docs/DECODER_PROGRAM.md` rule 6).
    """
    panel = float(panel_spectra) if panel_spectra else None
    cap = _as_float(env.get("MARLIN_MAX_SPECTRA"))
    shards = _as_float(env.get("MARLIN_SHARDS")) or 1.0
    if cap is None:
        return panel
    truncated = cap * shards
    return min(truncated, panel) if panel else truncated


def evaluation_progress(snapshot: RunSnapshot) -> Progress:
    """Decoded rows out of the panel size.

    The predictions file wins when it is reachable, because it is the artifact
    the metrics are scored from; otherwise the heartbeat is the only signal.
    """
    total = evaluation_target_rows(snapshot.panel_spectra, snapshot.env)
    rows: float | None = None
    source = "none"
    if snapshot.partial_metrics and snapshot.partial_metrics.get("total_spectra"):
        rows = float(snapshot.partial_metrics["total_spectra"])
        source = "predictions file"
    elif snapshot.heartbeats:
        rows = float(snapshot.heartbeats[-1].rows)
        source = "console heartbeat"
    else:
        merged = MERGED_ROWS_RE.search("\n".join(snapshot.console))
        if merged:
            rows = float(merged["rows"])
            source = "console merge line"

    secondary = None
    if source == "predictions file" and snapshot.heartbeats:
        beat_rows = float(snapshot.heartbeats[-1].rows)
        if rows is not None and beat_rows != rows:
            secondary = ("console heartbeat", beat_rows)

    rate = _heartbeat_rows_per_hour(snapshot.heartbeats)
    eta = None
    if rate and rows is not None and total:
        remaining = max(total - rows, 0.0)
        eta = remaining / rate * 3600.0 if rate > 0 else None
    return Progress(
        unit="rows",
        current=rows,
        total=total,
        source=source,
        rate_per_hour=rate,
        eta_seconds=eta,
        secondary=secondary,
    )


def _heartbeat_rows_per_hour(beats: Sequence[Heartbeat]) -> float | None:
    """Steady-state throughput: the whole heartbeat window, first beat dropped.

    The first beat covers process startup - pip install, checkpoint hashing,
    model load - so including it understates the decode rate.
    """
    usable = list(beats[1:]) if len(beats) > 2 else list(beats)
    if len(usable) < 2:
        return None
    span = usable[-1].elapsed_seconds - usable[0].elapsed_seconds
    if span <= 0:
        return None
    return (usable[-1].rows - usable[0].rows) / span * 3600.0


def run_progress(snapshot: RunSnapshot) -> Progress:
    if snapshot.kind == TRAINING:
        return training_progress(snapshot)
    if snapshot.kind == EVALUATION:
        return evaluation_progress(snapshot)
    return Progress(unit="", current=None, total=None, source="none")


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# partial predictions


def find_predictions(run_name: str | None, roots: Sequence[Path]) -> Path | None:
    """The merged predictions file of a run directory, if any root holds one."""
    if not run_name:
        return None
    for root in roots:
        merged = root / run_name / "predictions.jsonl"
        if merged.is_file():
            return merged
    return None


def find_shard_predictions(run_name: str | None, roots: Sequence[Path]) -> list[Path]:
    """Per-shard predictions files; a run only merges them when it finishes."""
    if not run_name:
        return []
    for root in roots:
        run_dir = root / run_name
        if not run_dir.is_dir():
            continue
        shards = sorted(run_dir.glob("shard*/predictions.jsonl"))
        if shards:
            return shards
    return []


def read_prediction_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load prediction rows, keeping the first row per spectrum.

    Shards are interleaved slices of one panel, so concatenating them is
    correct; the de-duplication is there only so a merged file read alongside
    its own shards cannot double-count a spectrum.
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        try:
            text = path.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A row being appended right now can be a half-written line.
                continue
            key = str(row.get("spec_name"))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def partial_metrics_for(
    run_name: str | None, roots: Sequence[Path]
) -> tuple[dict[str, Any] | None, Path | None, str]:
    """Score whatever rows exist so far, or explain why nothing was scored."""
    if not run_name:
        return None, None, "no run name in the task container arguments"
    merged = find_predictions(run_name, roots)
    shards = find_shard_predictions(run_name, roots)
    paths = ([merged] if merged else []) + shards
    if not paths:
        return (
            None,
            None,
            "predictions file not reachable from this host "
            f"(looked for {run_name}/predictions.jsonl under "
            f"{', '.join(str(root) for root in roots)})",
        )
    rows = read_prediction_rows(paths)
    if not rows:
        return None, paths[0], "predictions file present but empty"
    return frigid_convention_metrics(rows), paths[0], ""


# --------------------------------------------------------------------------
# health


def check_health(
    snapshot: RunSnapshot,
    progress: Progress,
    workers: dict[str, WorkerInfo],
    queue_worker_counts: dict[str, int],
    now: datetime,
    previous: dict[str, Any] | None,
    stall_minutes: float = 30.0,
    disk_warn_percent: float = 20.0,
    disk_critical_percent: float = 10.0,
    worker_silence_minutes: float = 10.0,
) -> list[Flag]:
    """Every failure mode this project has already paid for, in one place."""
    flags: list[Flag] = []
    stall = timedelta(minutes=stall_minutes)

    if snapshot.status == "queued":
        listening = queue_worker_counts.get(snapshot.queue_id or "", 0)
        if listening == 0:
            flags.append(
                Flag(
                    CRITICAL,
                    "queue-has-no-worker",
                    f"queued on {snapshot.queue_name or snapshot.queue_id!r} "
                    "and no worker is serving that queue - this run will never start",
                )
            )
        else:
            position = (
                f"position {snapshot.queue_position}"
                if snapshot.queue_position is not None
                else "in queue"
            )
            flags.append(
                Flag(
                    INFO,
                    "queued-behind",
                    f"{position} on {snapshot.queue_name}, {listening} worker(s) serving it",
                )
            )

    if snapshot.status == "in_progress":
        worker = workers.get(snapshot.worker or "")
        if not snapshot.worker:
            flags.append(
                Flag(
                    CRITICAL,
                    "no-worker-attached",
                    "reported as running with no worker recorded on the task",
                )
            )
        elif worker is None:
            flags.append(
                Flag(
                    WARNING,
                    "no-worker-attached",
                    f"worker {snapshot.worker} is not registered with the server any more",
                )
            )
        elif worker.last_activity and now - worker.last_activity > timedelta(
            minutes=worker_silence_minutes
        ):
            flags.append(
                Flag(
                    CRITICAL,
                    "no-worker-attached",
                    f"worker {snapshot.worker} last reported "
                    f"{_duration(now - worker.last_activity)} ago",
                )
            )

        stalled = _stall_flag(snapshot, progress, now, previous, stall)
        if stalled:
            flags.append(stalled)

        if snapshot.last_update and now - snapshot.last_update > stall:
            flags.append(
                Flag(
                    WARNING,
                    "no-recent-report",
                    f"nothing reported to ClearML for {_duration(now - snapshot.last_update)}",
                )
            )

    if snapshot.kind == TRAINING and snapshot.status == "in_progress":
        flags.extend(check_loss(snapshot.scalars))

    if snapshot.status == "in_progress":
        # Only the worker this run is actually on. The fleet-wide disk state is
        # reported once in its own section rather than repeated per run.
        flags.extend(
            check_worker_disk(
                {snapshot.worker: workers[snapshot.worker]}
                if snapshot.worker in workers
                else {},
                disk_warn_percent,
                disk_critical_percent,
            )
        )

    flags.sort(key=lambda flag: _LEVEL_ORDER[flag.level])
    return flags


def check_worker_disk(
    workers: dict[str, WorkerInfo],
    disk_warn_percent: float = 20.0,
    disk_critical_percent: float = 10.0,
) -> list[Flag]:
    """A worker that runs out of disk loses the run and the artifacts with it.

    `docs/DECODER_PROGRAM.md` §8 records aiagent03's local ext4 at 89% full
    while it carried the only copy of the control-r2 checkpoint, so this is a
    measured failure surface, not a precaution.
    """
    flags: list[Flag] = []
    for worker_id, worker in sorted(workers.items()):
        free = worker.disk_free_percent
        if free is None:
            continue
        if free < disk_critical_percent:
            flags.append(
                Flag(
                    CRITICAL,
                    "worker-disk-low",
                    f"{worker_id} has {free:.1f}% disk free",
                )
            )
        elif free < disk_warn_percent:
            flags.append(
                Flag(
                    WARNING, "worker-disk-low", f"{worker_id} has {free:.1f}% disk free"
                )
            )
    return flags


def _stall_flag(
    snapshot: RunSnapshot,
    progress: Progress,
    now: datetime,
    previous: dict[str, Any] | None,
    stall: timedelta,
) -> Flag | None:
    """Progress that has not moved for `stall`, judged in-band where possible.

    An evaluation prints its row count with a timestamp every ten minutes, so
    two heartbeats far enough apart at the same row count settle the question
    inside a single invocation. Training arms have no such stamp, so they fall
    back to the counter this script persisted on its previous run.
    """
    if progress.current is None:
        return None

    if snapshot.heartbeats:
        latest = snapshot.heartbeats[-1]
        unchanged_since = latest
        for beat in reversed(snapshot.heartbeats):
            if beat.rows != latest.rows:
                break
            unchanged_since = beat
        idle = timedelta(
            seconds=latest.elapsed_seconds - unchanged_since.elapsed_seconds
        )
        if idle >= stall:
            return Flag(
                CRITICAL,
                "progress-stalled",
                f"rows stuck at {latest.rows} for {_duration(idle)} of heartbeats",
            )

    if previous and previous.get("unit") == progress.unit:
        prior = previous.get("current")
        seen_at = _parse_iso(previous.get("first_seen_at"))
        if prior is not None and seen_at and float(prior) >= float(progress.current):
            idle = now - seen_at
            if idle >= stall:
                return Flag(
                    CRITICAL,
                    "progress-stalled",
                    f"{progress.unit} stuck at {progress.current:g} for {_duration(idle)}",
                )
    return None


def check_loss(scalars: Sequence[Series], flat_window: int = 5) -> list[Flag]:
    """NaN kills a run silently; a flat loss means it is not learning."""
    flags: list[Flag] = []
    for series in scalars:
        if "loss" not in series.name:
            continue
        values = [v for v in series.y if v is not None]
        if not values:
            continue
        if not _is_finite(values[-1]):
            flags.append(
                Flag(
                    CRITICAL,
                    "loss-nan",
                    f"{series.label} is {values[-1]} at step {series.x[-1]:g}",
                )
            )
            continue
        tail = [float(v) for v in values[-flat_window:] if _is_finite(v)]
        if len(tail) >= flat_window:
            spread = max(tail) - min(tail)
            scale = max(abs(sum(tail) / len(tail)), 1e-12)
            if spread / scale < 1e-6:
                flags.append(
                    Flag(
                        WARNING,
                        "loss-flat",
                        f"{series.label} moved {spread:.3g} over its last "
                        f"{len(tail)} reports (value {tail[-1]:.6g})",
                    )
                )
    return flags


# --------------------------------------------------------------------------
# ClearML access


class ClearMLSource:
    """Thin read-only wrapper over the ClearML API.

    Everything above this line is pure, so the report and the health rules are
    tested without a server.
    """

    def __init__(self, project: str = DEFAULT_PROJECT) -> None:
        from clearml.backend_api.session.client import APIClient

        self._client = APIClient()
        self._project = project
        self._project_id: str | None = None
        self._queue_cache: dict[str, Any] = {}

    @property
    def project_id(self) -> str:
        if self._project_id is None:
            escaped = re.escape(self._project)
            projects = self._client.projects.get_all(name=f"^{escaped}$")
            if not projects:
                raise SystemExit(f"ClearML project not found: {self._project!r}")
            self._project_id = projects[0].id
        return self._project_id

    def list_tasks(self, page_size: int = 200) -> list[Any]:
        return list(
            self._client.tasks.get_all(
                project=[self.project_id],
                order_by=["-last_update"],
                page=0,
                page_size=page_size,
                only_fields=[
                    "id",
                    "name",
                    "status",
                    "status_message",
                    "created",
                    "started",
                    "completed",
                    "last_update",
                    "last_worker",
                    "active_duration",
                    "execution.queue",
                    "container",
                    "script.entry_point",
                ],
            )
        )

    def get_task(self, task_id: str) -> Any:
        tasks = self._client.tasks.get_all(id=[task_id])
        if not tasks:
            raise SystemExit(f"ClearML task not found: {task_id}")
        return tasks[0]

    def queue(self, queue_id: str) -> Any | None:
        if queue_id not in self._queue_cache:
            try:
                self._queue_cache[queue_id] = self._client.queues.get_by_id(queue_id)
            except Exception:  # noqa: BLE001 - a deleted queue must not kill the report
                self._queue_cache[queue_id] = None
        return self._queue_cache[queue_id]

    def workers(self) -> list[Any]:
        return list(self._client.workers.get_all())

    def worker_stats(
        self, worker_ids: Sequence[str], window_seconds: int = 1800
    ) -> dict[str, dict[str, float]]:
        if not worker_ids:
            return {}
        now = int(time.time())
        try:
            response = self._client.workers.get_stats(
                worker_ids=list(worker_ids),
                from_date=now - window_seconds,
                to_date=now,
                interval=window_seconds,
                items=[
                    {"key": "disk_free_home", "aggregation": "avg"},
                    {"key": "gpu_usage", "aggregation": "avg"},
                ],
            )
        except Exception:  # noqa: BLE001 - stats are advisory, the report is not
            return {}
        return parse_worker_stats(
            response.to_dict() if hasattr(response, "to_dict") else response
        )

    def scalars(self, task_id: str, max_samples: int) -> list[Series]:
        from clearml import Task

        task = Task.get_task(task_id=task_id)
        reported = task.get_reported_scalars(max_samples=max_samples) or {}
        series: list[Series] = []
        for title, by_name in sorted(reported.items()):
            for name, values in sorted(by_name.items()):
                series.append(
                    Series(
                        title=title,
                        name=name,
                        x=list(values.get("x", [])),
                        y=list(values.get("y", [])),
                    )
                )
        return series

    def console(self, task_id: str, blocks: int) -> list[str]:
        from clearml import Task

        task = Task.get_task(task_id=task_id)
        return list(task.get_reported_console_output(number_of_reports=blocks) or [])

    def script_diff(self, task_id: str) -> str:
        from clearml import Task

        task = Task.get_task(task_id=task_id)
        script = task.data.script
        return getattr(script, "diff", "") or ""

    def task_url(self, task_id: str) -> str:
        from clearml import Task

        return Task.get_task(task_id=task_id).get_output_log_web_page()


def parse_worker_stats(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Reduce ``workers.get_stats`` to the last value of each metric."""
    result: dict[str, dict[str, float]] = {}
    for worker in payload.get("workers", []) or []:
        worker_id = worker.get("worker")
        if not worker_id:
            continue
        metrics: dict[str, float] = {}
        for metric in worker.get("metrics", []) or []:
            key = metric.get("metric")
            for stat in metric.get("stats", []) or []:
                values = [v for v in (stat.get("values") or []) if v is not None]
                if key and values:
                    metrics[key] = float(values[-1])
        if metrics:
            result[worker_id] = metrics
    return result


def collect_workers(
    source: ClearMLSource,
) -> tuple[dict[str, WorkerInfo], dict[str, int]]:
    """Registered workers, and how many of them serve each queue by name."""
    workers: dict[str, WorkerInfo] = {}
    queue_counts_by_name: dict[str, int] = {}
    for raw in source.workers():
        queues = [
            q.name
            for q in (getattr(raw, "queues", None) or [])
            if getattr(q, "name", None)
        ]
        task = getattr(raw, "task", None)
        workers[raw.id] = WorkerInfo(
            worker_id=raw.id,
            queues=queues,
            task_id=getattr(task, "id", None) if task else None,
            last_activity=_coerce_datetime(getattr(raw, "last_activity_time", None)),
        )
        for queue_name in queues:
            queue_counts_by_name[queue_name] = (
                queue_counts_by_name.get(queue_name, 0) + 1
            )

    stats = source.worker_stats(list(workers))
    for worker_id, metrics in stats.items():
        if worker_id in workers:
            workers[worker_id].disk_free_percent = metrics.get("disk_free_home")
            workers[worker_id].gpu_usage_percent = metrics.get("gpu_usage")
    return workers, queue_counts_by_name


def build_snapshot(
    source: ClearMLSource,
    raw_task: Any,
    evaluation_roots: Sequence[Path],
    console_blocks: int,
    scalar_samples: int,
) -> RunSnapshot:
    container = getattr(raw_task, "container", None) or {}
    if not isinstance(container, dict):
        container = container.to_dict() if hasattr(container, "to_dict") else {}
    env = parse_container_env(container.get("arguments"))
    script = getattr(raw_task, "script", None)
    entry_point = getattr(script, "entry_point", "") if script else ""
    execution = getattr(raw_task, "execution", None)
    queue_id = getattr(execution, "queue", None) if execution else None

    snapshot = RunSnapshot(
        task_id=raw_task.id,
        name=raw_task.name,
        status=_status_text(raw_task.status),
        status_message=getattr(raw_task, "status_message", "") or "",
        kind=classify_run(entry_point, env, raw_task.name),
        queue_id=queue_id,
        entry_point=entry_point,
        env=env,
        worker=getattr(raw_task, "last_worker", None) or None,
        started=_coerce_datetime(getattr(raw_task, "started", None)),
        last_update=_coerce_datetime(getattr(raw_task, "last_update", None)),
        active_duration=_as_float(getattr(raw_task, "active_duration", None)),
    )

    if queue_id:
        queue = source.queue(queue_id)
        if queue is not None:
            snapshot.queue_name = queue.name
            entries = [entry.task for entry in (queue.entries or [])]
            if snapshot.task_id in entries:
                snapshot.queue_position = entries.index(snapshot.task_id) + 1

    snapshot.scalars = source.scalars(snapshot.task_id, scalar_samples)
    snapshot.console = source.console(snapshot.task_id, console_blocks)
    snapshot.url = source.task_url(snapshot.task_id)

    if snapshot.kind == EVALUATION:
        snapshot.heartbeats = parse_heartbeats("\n".join(snapshot.console))
        snapshot.panel_path = parse_panel_path(source.script_diff(snapshot.task_id))
        snapshot.panel_spectra = panel_spectra_count(snapshot.panel_path)
        metrics, path, note = partial_metrics_for(
            env.get("MARLIN_RUN_NAME"), evaluation_roots
        )
        snapshot.partial_metrics = metrics
        snapshot.predictions_path = path
        snapshot.predictions_note = note
        if path is not None and path.exists():
            snapshot.predictions_mtime = datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            )
    return snapshot


def _status_text(value: Any) -> str:
    """ClearML returns statuses as an enum whose members do not sort or compare.

    Every rule in this file is written against the plain strings ClearML's own
    UI shows, so the enum is flattened once, here, at the boundary.
    """
    if value is None:
        return ""
    return str(getattr(value, "value", value))


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return _parse_iso(str(value))


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# state, for the stall check across invocations


def load_state(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def update_state(
    state: dict[str, Any], task_id: str, progress: Progress, now: datetime
) -> dict[str, Any]:
    """Remember when a counter first reached its current value.

    Storing the *first* sighting rather than the latest is what makes "has not
    advanced in 30 minutes" answerable at all: a run that keeps reporting the
    same number must not look fresh.
    """
    entry = state.get(task_id)
    if (
        entry
        and entry.get("unit") == progress.unit
        and progress.current is not None
        and entry.get("current") is not None
        and float(entry["current"]) >= float(progress.current)
    ):
        entry["last_seen_at"] = now.isoformat()
        state[task_id] = entry
        return state
    state[task_id] = {
        "unit": progress.unit,
        "current": progress.current,
        "first_seen_at": now.isoformat(),
        "last_seen_at": now.isoformat(),
    }
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    except OSError:
        pass


# --------------------------------------------------------------------------
# rendering


class Palette:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def __call__(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def level(self, level: str, text: str) -> str:
        return self(text, {CRITICAL: "1;31", WARNING: "1;33", INFO: "0;36"}[level])

    def status(self, status: str, text: str) -> str:
        code = {
            "in_progress": "1;32",
            "queued": "1;33",
            "failed": "1;31",
            "stopped": "0;35",
            "completed": "0;36",
        }.get(status, "0")
        return self(text, code)


def _duration(delta: timedelta | float | None) -> str:
    if delta is None:
        return "-"
    seconds = delta.total_seconds() if isinstance(delta, timedelta) else float(delta)
    if seconds < 0:
        return "-"
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def _number(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "-"
    if not _is_finite(value):
        return str(value)
    return f"{float(value):.{digits}g}"


def render_progress(progress: Progress) -> str:
    if progress.current is None:
        return f"progress: unknown ({progress.source})"
    total = f"/{progress.total:g}" if progress.total else ""
    fraction = progress.fraction
    bar = ""
    if fraction is not None:
        filled = int(round(min(fraction, 1.0) * 24))
        bar = f" [{'#' * filled}{'.' * (24 - filled)}] {fraction * 100:5.1f}%"
    rate = (
        f"  {progress.rate_per_hour:.1f} {progress.unit}/h"
        if progress.rate_per_hour
        else ""
    )
    eta = f"  ETA {_duration(progress.eta_seconds)}" if progress.eta_seconds else ""
    second = (
        f", {progress.secondary[0]} says {progress.secondary[1]:g}"
        if progress.secondary
        else ""
    )
    return (
        f"progress: {progress.current:g}{total} {progress.unit}{bar}{rate}{eta}"
        f"   (via {progress.source}{second})"
    )


def render_run(
    snapshot: RunSnapshot,
    progress: Progress,
    flags: Sequence[Flag],
    workers: dict[str, WorkerInfo],
    now: datetime,
    palette: Palette,
    console_lines: int,
    scalar_limit: int,
) -> str:
    lines: list[str] = []
    head = (
        f"{palette(snapshot.name, '1')}  "
        f"[{palette.status(snapshot.status, snapshot.status)}] "
        f"{snapshot.kind}"
    )
    lines.append(head)
    lines.append(f"  task {snapshot.task_id}   {snapshot.url or ''}")

    worker = workers.get(snapshot.worker or "")
    worker_text = snapshot.worker or "-"
    if worker and worker.disk_free_percent is not None:
        worker_text += f" (disk free {worker.disk_free_percent:.1f}%"
        if worker.gpu_usage_percent is not None:
            worker_text += f", gpu {worker.gpu_usage_percent:.0f}%"
        worker_text += ")"
    elapsed = None
    if snapshot.active_duration:
        elapsed = snapshot.active_duration
    elif snapshot.started:
        elapsed = (now - snapshot.started).total_seconds()
    queue_text = snapshot.queue_name or snapshot.queue_id or "-"
    if snapshot.queue_position is not None:
        queue_text += f" (#{snapshot.queue_position})"
    lines.append(
        f"  queue {queue_text}   worker {worker_text}   elapsed {_duration(elapsed)}"
        f"   last report {_duration(now - snapshot.last_update) if snapshot.last_update else '-'} ago"
    )
    lines.append("  " + render_progress(progress))

    if snapshot.kind == EVALUATION:
        lines.append("  " + render_partial_metrics(snapshot, palette))

    if snapshot.scalars:
        lines.append("  scalars:")
        for series in rank_series(snapshot.scalars)[:scalar_limit]:
            summary = summarise_series(series)
            arrow = {"up": "^", "down": "v", "flat": "="}.get(
                summary.get("direction", ""), " "
            )
            delta = summary.get("delta")
            delta_text = (
                f"{arrow} {_number(delta, 3)} over {summary.get('window')} pts"
                if delta is not None
                else ""
            )
            lines.append(
                f"    {summary['label']:<58} {_number(summary['last']):>14}"
                f" @ step {_number(summary.get('last_step'), 8):<8} {delta_text}"
            )
        if len(snapshot.scalars) > scalar_limit:
            lines.append(
                f"    ... {len(snapshot.scalars) - scalar_limit} more series "
                f"(--scalar-limit 0 for all)"
            )
    elif snapshot.kind == TRAINING:
        lines.append("  scalars: none reported yet")

    tail = console_tail(snapshot.console, console_lines)
    if tail:
        lines.append("  console tail:")
        lines.extend(f"    | {line}" for line in tail)

    if flags:
        lines.append("  health:")
        for flag in flags:
            lines.append(
                f"    {palette.level(flag.level, flag.level.ljust(8))} "
                f"{flag.code}: {flag.message}"
            )
    else:
        lines.append("  health: " + palette("ok", "0;32"))
    return "\n".join(lines)


def render_partial_metrics(snapshot: RunSnapshot, palette: Palette) -> str:
    metrics = snapshot.partial_metrics
    if not metrics:
        return f"partial metrics: unavailable - {snapshot.predictions_note}"
    complete = (
        snapshot.panel_spectra is not None
        and metrics.get("total_spectra") == snapshot.panel_spectra
        and snapshot.status not in ("in_progress", "queued")
    )
    label = "FINAL" if complete else palette("PARTIAL", "1;33")
    return (
        f"{label} over {metrics['total_spectra']} rows"
        f"{f'/{snapshot.panel_spectra}' if snapshot.panel_spectra else ''}: "
        f"Exact@1 {_percent(metrics.get('exact_match_top1'))}, "
        f"Exact@10 {_percent(metrics.get('exact_match_top10'))}, "
        f"return rate {_percent(metrics.get('candidate_return_rate'))}, "
        f"Tanimoto@1 {_number(metrics.get('tanimoto_top1_mean'), 4)}, "
        f"truncated {metrics.get('truncated_spectra')}"
    )


def console_tail(blocks: Sequence[str], limit: int) -> list[str]:
    if limit <= 0:
        return []
    text = "\n".join(blocks)
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return lines[-limit:]


def render_workers(
    workers: dict[str, WorkerInfo],
    queues_of_interest: set[str],
    worker_flags: Sequence[Flag],
    palette: Palette,
    now: datetime,
) -> str:
    lines = [
        "workers serving " + (", ".join(sorted(queues_of_interest)) or "no queue") + ":"
    ]
    relevant = {
        worker_id: worker
        for worker_id, worker in workers.items()
        if queues_of_interest & set(worker.queues)
    } or workers
    for worker_id, worker in sorted(relevant.items()):
        idle = (
            f"idle {_duration(now - worker.last_activity)}"
            if worker.last_activity
            else "last activity unknown"
        )
        lines.append(
            f"  {worker_id:<24} queues {','.join(worker.queues) or '-':<28} "
            f"task {worker.task_id or '-':<34} "
            f"disk free {_number(worker.disk_free_percent, 3):>6}%  "
            f"gpu {_number(worker.gpu_usage_percent, 3):>6}%  {idle} since report"
        )
    for flag in worker_flags:
        lines.append(
            f"  {palette.level(flag.level, flag.level.ljust(8))} {flag.code}: {flag.message}"
        )
    return "\n".join(lines)


def render_summary(
    results: Sequence[tuple[RunSnapshot, Progress, list[Flag]]],
    palette: Palette,
    now: datetime,
    extra_flags: Sequence[Flag] = (),
) -> str:
    counts: dict[str, int] = {}
    for _, _, flags in results:
        for flag in flags:
            counts[flag.level] = counts.get(flag.level, 0) + 1
    for flag in extra_flags:
        counts[flag.level] = counts.get(flag.level, 0) + 1
    by_status: dict[str, int] = {}
    for snapshot, _, _ in results:
        by_status[snapshot.status] = by_status.get(snapshot.status, 0) + 1
    status_text = ", ".join(f"{n} {status}" for status, n in sorted(by_status.items()))
    health_text = ", ".join(
        palette.level(level, f"{counts[level]} {level.lower()}")
        for level in (CRITICAL, WARNING, INFO)
        if level in counts
    )
    return (
        f"{len(results)} run(s): {status_text or 'none'}"
        f"   health: {health_text or palette('all ok', '0;32')}"
        f"   at {now.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )


def snapshot_to_dict(
    snapshot: RunSnapshot, progress: Progress, flags: Sequence[Flag], console_lines: int
) -> dict[str, Any]:
    return {
        "task_id": snapshot.task_id,
        "name": snapshot.name,
        "status": snapshot.status,
        "kind": snapshot.kind,
        "queue": snapshot.queue_name,
        "queue_position": snapshot.queue_position,
        "worker": snapshot.worker,
        "elapsed_seconds": snapshot.active_duration,
        "started": snapshot.started.isoformat() if snapshot.started else None,
        "last_update": snapshot.last_update.isoformat()
        if snapshot.last_update
        else None,
        "url": snapshot.url,
        "progress": {
            "unit": progress.unit,
            "current": progress.current,
            "total": progress.total,
            "fraction": progress.fraction,
            "source": progress.source,
            "rate_per_hour": progress.rate_per_hour,
            "eta_seconds": progress.eta_seconds,
        },
        "scalars": [
            summarise_series(series) for series in rank_series(snapshot.scalars)
        ],
        "console_tail": console_tail(snapshot.console, console_lines),
        "panel": {"path": snapshot.panel_path, "spectra": snapshot.panel_spectra},
        "partial_metrics": snapshot.partial_metrics,
        "partial_metrics_note": snapshot.predictions_note,
        "predictions_path": str(snapshot.predictions_path)
        if snapshot.predictions_path
        else None,
        "health": [
            {"level": flag.level, "code": flag.code, "message": flag.message}
            for flag in flags
        ],
    }


# --------------------------------------------------------------------------
# entry point


def select_tasks(
    raw_tasks: Iterable[Any],
    statuses: Sequence[str] | None,
    since_hours: float | None,
    now: datetime,
    task_ids: Sequence[str] = (),
) -> list[Any]:
    """Which runs the dashboard is about.

    Default is everything active plus anything that changed recently, because a
    run that failed twenty minutes ago is exactly what a health dashboard is
    for.
    """
    if task_ids:
        wanted = set(task_ids)
        return [task for task in raw_tasks if task.id in wanted]
    selected = []
    cutoff = now - timedelta(hours=since_hours) if since_hours else None
    for task in raw_tasks:
        status = _status_text(task.status)
        if statuses and status not in statuses:
            continue
        if statuses:
            selected.append(task)
            continue
        if status in ACTIVE_STATUSES:
            selected.append(task)
            continue
        last_update = _coerce_datetime(getattr(task, "last_update", None))
        if cutoff and last_update and last_update >= cutoff:
            selected.append(task)
    return selected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Show exactly these tasks, whatever their age or status.",
    )
    parser.add_argument(
        "--status",
        action="append",
        default=[],
        help="Restrict to these ClearML statuses; repeatable.",
    )
    parser.add_argument(
        "--since-hours",
        type=float,
        default=12.0,
        help="Also show finished runs updated within this window (default 12).",
    )
    parser.add_argument("--console-lines", type=int, default=6)
    parser.add_argument("--console-blocks", type=int, default=12)
    parser.add_argument("--scalar-samples", type=int, default=200)
    parser.add_argument(
        "--scalar-limit",
        type=int,
        default=12,
        help="Scalar series printed per run; 0 for all.",
    )
    parser.add_argument(
        "--evaluation-root",
        action="append",
        default=[],
        help=f"Where evaluation run directories live (default {DEFAULT_EVALUATION_ROOTS[0]}).",
    )
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--stall-minutes", type=float, default=30.0)
    parser.add_argument("--disk-warn-percent", type=float, default=20.0)
    parser.add_argument("--disk-critical-percent", type=float, default=10.0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument(
        "--watch",
        type=float,
        default=0.0,
        help="Refresh every N seconds instead of printing once.",
    )
    return parser.parse_args(argv)


def report_once(args: argparse.Namespace, source: ClearMLSource) -> tuple[str, int]:
    now = datetime.now(timezone.utc)
    roots = [Path(root) for root in (args.evaluation_root or DEFAULT_EVALUATION_ROOTS)]
    raw_tasks = (
        [source.get_task(task_id) for task_id in args.task_id]
        if args.task_id
        else source.list_tasks()
    )
    selected = select_tasks(
        raw_tasks, args.status or None, args.since_hours, now, args.task_id
    )
    workers, queue_counts_by_name = collect_workers(source)

    state = load_state(args.state_file)
    results: list[tuple[RunSnapshot, Progress, list[Flag]]] = []
    for raw_task in selected:
        snapshot = build_snapshot(
            source, raw_task, roots, args.console_blocks, args.scalar_samples
        )
        progress = run_progress(snapshot)
        queue_counts = {
            snapshot.queue_id or "": queue_counts_by_name.get(
                snapshot.queue_name or "", 0
            )
        }
        flags = check_health(
            snapshot,
            progress,
            workers,
            queue_counts,
            now,
            state.get(snapshot.task_id),
            stall_minutes=args.stall_minutes,
            disk_warn_percent=args.disk_warn_percent,
            disk_critical_percent=args.disk_critical_percent,
        )
        state = update_state(state, snapshot.task_id, progress, now)
        results.append((snapshot, progress, flags))
    save_state(args.state_file, state)

    queues_of_interest = {
        snapshot.queue_name for snapshot, _, _ in results if snapshot.queue_name
    }
    fleet = {
        worker_id: worker
        for worker_id, worker in workers.items()
        if queues_of_interest & set(worker.queues)
    }
    worker_flags = check_worker_disk(
        fleet, args.disk_warn_percent, args.disk_critical_percent
    )

    worst = min(
        (
            _LEVEL_ORDER[flag.level]
            for flag in [flag for _, _, flags in results for flag in flags]
            + list(worker_flags)
        ),
        default=99,
    )
    exit_code = 2 if worst == 0 else (1 if worst == 1 else 0)

    if args.json:
        payload = {
            "generated_at": now.isoformat(),
            "project": args.project,
            "runs": [
                snapshot_to_dict(snapshot, progress, flags, args.console_lines)
                for snapshot, progress, flags in results
            ],
            "workers": [
                {
                    "worker": worker.worker_id,
                    "queues": worker.queues,
                    "task": worker.task_id,
                    "disk_free_percent": worker.disk_free_percent,
                    "gpu_usage_percent": worker.gpu_usage_percent,
                    "last_activity": worker.last_activity.isoformat()
                    if worker.last_activity
                    else None,
                }
                for worker in sorted(workers.values(), key=lambda w: w.worker_id)
            ],
            "worker_health": [
                {"level": flag.level, "code": flag.code, "message": flag.message}
                for flag in worker_flags
            ],
        }
        return json.dumps(payload, indent=2, sort_keys=True, default=str), exit_code

    palette = Palette(not args.no_color and sys.stdout.isatty())
    scalar_limit = args.scalar_limit if args.scalar_limit > 0 else 10**6
    blocks = [
        render_summary(results, palette, now, worker_flags),
        "",
        render_workers(workers, queues_of_interest, worker_flags, palette, now),
        "",
    ]
    ordered = sorted(
        results,
        key=lambda item: (
            0
            if item[0].status == "in_progress"
            else 1
            if item[0].status == "queued"
            else 2,
            item[0].name,
        ),
    )
    for snapshot, progress, flags in ordered:
        blocks.append(
            render_run(
                snapshot,
                progress,
                flags,
                workers,
                now,
                palette,
                args.console_lines,
                scalar_limit,
            )
        )
        blocks.append("")
    return "\n".join(blocks), exit_code


def main(argv: Sequence[str] | None = None) -> int:
    if os.environ.get("LD_PRELOAD"):
        # *.innopolis.university looks unreachable under this preload; a
        # dashboard that reports "server down" because of it is worse than
        # useless, so drop it and run again.
        env = dict(os.environ)
        env.pop("LD_PRELOAD", None)
        os.execve(
            sys.executable,
            [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:])],
            env,
        )

    args = parse_args(argv)
    source = ClearMLSource(args.project)
    if args.watch <= 0:
        text, code = report_once(args, source)
        print(text)
        return code
    while True:
        text, code = report_once(args, source)
        print("\033[2J\033[H" + text, flush=True)
        del code
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
