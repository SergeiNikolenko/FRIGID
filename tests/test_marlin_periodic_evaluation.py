import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from marlin.periodic_evaluation import PeriodicMolecularEvaluation
import marlin.periodic_evaluation as periodic_evaluation


def test_periodic_evaluation_waits_for_current_run_checkpoint(tmp_path: Path) -> None:
    config = OmegaConf.create(
        {
            "evaluation": {"enabled": True, "interval_steps": 1000},
            "output": {
                "root": str(tmp_path),
                "checkpoints": str(tmp_path / "checkpoints"),
                "checkpoint_interval": 1000,
            },
        }
    )
    callback = PeriodicMolecularEvaluation(config, project_root=tmp_path)
    calls = []
    callback._run = calls.append
    trainer = SimpleNamespace(is_global_zero=True, global_step=20000)

    callback.on_train_batch_end(trainer, None, None, None, 0)
    assert calls == []

    checkpoint = tmp_path / "checkpoints/step=20000.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    callback.on_train_batch_end(trainer, None, None, None, 0)
    assert calls == [20000]


def test_disabled_periodic_evaluation_allows_different_checkpoint_interval(
    tmp_path: Path,
) -> None:
    config = OmegaConf.create(
        {
            "evaluation": {"enabled": False, "interval_steps": 5000},
            "output": {
                "root": str(tmp_path),
                "checkpoints": str(tmp_path / "checkpoints"),
                "checkpoint_interval": 200,
            },
        }
    )

    PeriodicMolecularEvaluation(config, project_root=tmp_path)


def test_periodic_evaluation_retries_after_final_checkpoint(tmp_path: Path) -> None:
    config = OmegaConf.create(
        {
            "evaluation": {"enabled": True, "interval_steps": 1000},
            "output": {
                "root": str(tmp_path),
                "checkpoints": str(tmp_path / "checkpoints"),
                "checkpoint_interval": 1000,
            },
        }
    )
    callback = PeriodicMolecularEvaluation(config, project_root=tmp_path)
    calls = []
    callback._run = calls.append
    trainer = SimpleNamespace(is_global_zero=True, global_step=1000)

    callback.on_train_batch_end(trainer, None, None, None, 0)
    checkpoint = tmp_path / "checkpoints/step=1000.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    callback.on_train_end(trainer, None)

    assert calls == [1000]


def test_periodic_evaluation_runs_completed_boundary_on_next_step(
    tmp_path: Path,
) -> None:
    config = OmegaConf.create(
        {
            "evaluation": {"enabled": True, "interval_steps": 500},
            "output": {
                "root": str(tmp_path),
                "checkpoints": str(tmp_path / "checkpoints"),
                "checkpoint_interval": 500,
            },
        }
    )
    checkpoint = tmp_path / "checkpoints/step=500.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    callback = PeriodicMolecularEvaluation(config, project_root=tmp_path)
    calls = []
    callback._run = calls.append
    trainer = SimpleNamespace(is_global_zero=True, global_step=501)

    callback.on_train_batch_end(trainer, None, None, None, 0)
    callback.on_train_batch_end(trainer, None, None, None, 1)

    assert calls == [500]


def test_periodic_evaluation_does_not_repeat_failed_boundary(
    tmp_path: Path,
) -> None:
    config = OmegaConf.create(
        {
            "evaluation": {"enabled": True, "interval_steps": 500},
            "output": {
                "root": str(tmp_path),
                "checkpoints": str(tmp_path / "checkpoints"),
                "checkpoint_interval": 500,
            },
        }
    )
    checkpoint = tmp_path / "checkpoints/step=500.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    callback = PeriodicMolecularEvaluation(config, project_root=tmp_path)
    calls = []

    def fail(step: int) -> None:
        calls.append(step)
        raise RuntimeError("diagnostic failure")

    callback._run = fail
    callback._report_failure = lambda step, error: None
    trainer = SimpleNamespace(is_global_zero=True, global_step=501)

    callback.on_train_batch_end(trainer, None, None, None, 0)
    callback.on_train_batch_end(trainer, None, None, None, 1)

    assert calls == [500]


def test_periodic_evaluation_can_use_raw_weights(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = OmegaConf.create(
        {
            "data": {"tokenizer_file": str(tmp_path / "tokenizer.json")},
            "evaluation": {
                "enabled": True,
                "interval_steps": 500,
                "metadata": str(tmp_path / "metadata.csv"),
                "fingerprints": str(tmp_path / "dreams_predictions.npz"),
                "fingerprint_key": "probs",
                "use_ema": False,
                "lane": "dreams",
                "candidates": 16,
                "max_spectra": 4,
                "diversity_dropout": 0.3,
                "temperature": 1.0,
                "ppm_tolerance": 10.0,
                "seed": 42,
            },
            "output": {
                "root": str(tmp_path),
                "checkpoints": str(tmp_path / "checkpoints"),
                "checkpoint_interval": 500,
            },
        }
    )
    checkpoint = tmp_path / "checkpoints/step=500.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    commands = []
    monkeypatch.setattr(
        "marlin.periodic_evaluation.subprocess.run",
        lambda command, **kwargs: commands.append(command),
    )
    monkeypatch.setattr(
        "marlin.periodic_evaluation.torch.cuda.is_available",
        lambda: False,
    )

    PeriodicMolecularEvaluation(config, project_root=tmp_path)._run(500)

    assert "--no-ema" in commands[0]


def test_periodic_evaluation_forwards_fingerprint_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = OmegaConf.create(
        {
            "data": {"tokenizer_file": str(tmp_path / "tokenizer.json")},
            "evaluation": {
                "enabled": True,
                "interval_steps": 1000,
                "metadata": str(tmp_path / "metadata.csv"),
                "fingerprints": str(tmp_path / "dreams_predictions.npz"),
                "fingerprint_key": "probs",
                "threshold": 0.95,
                "lane": "dreams",
                "candidates": 16,
                "max_spectra": 4,
                "diversity_dropout": 0.3,
                "temperature": 1.0,
                "ppm_tolerance": 10.0,
                "seed": 42,
            },
            "output": {
                "root": str(tmp_path),
                "checkpoints": str(tmp_path / "checkpoints"),
                "checkpoint_interval": 1000,
            },
        }
    )
    checkpoint = tmp_path / "checkpoints/step=1000.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    commands = []
    monkeypatch.setattr(
        periodic_evaluation.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command),
    )
    monkeypatch.setattr(periodic_evaluation.torch.cuda, "is_available", lambda: False)

    PeriodicMolecularEvaluation(config, project_root=tmp_path)._run(1000)

    threshold_index = commands[0].index("--threshold")
    assert commands[0][threshold_index + 1] == "0.95"


def test_periodic_evaluation_forwards_locked_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = tmp_path / "micro32.tsv"
    config = OmegaConf.create(
        {
            "data": {"tokenizer_file": str(tmp_path / "tokenizer.json")},
            "evaluation": {
                "enabled": True,
                "interval_steps": 1000,
                "metadata": str(tmp_path / "metadata.csv"),
                "fingerprints": str(tmp_path / "dreams_predictions.npz"),
                "fingerprint_key": "probs",
                "spec_manifest": str(manifest),
                "lane": "dreams",
                "candidates": 16,
                "max_spectra": 32,
                "diversity_dropout": 0.3,
                "temperature": 1.0,
                "ppm_tolerance": 10.0,
                "seed": 42,
            },
            "output": {
                "root": str(tmp_path),
                "checkpoints": str(tmp_path / "checkpoints"),
                "checkpoint_interval": 1000,
            },
        }
    )
    checkpoint = tmp_path / "checkpoints/step=1000.ckpt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    commands = []
    monkeypatch.setattr(
        periodic_evaluation.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command),
    )
    monkeypatch.setattr(periodic_evaluation.torch.cuda, "is_available", lambda: False)

    PeriodicMolecularEvaluation(config, project_root=tmp_path)._run(1000)

    manifest_index = commands[0].index("--spec-manifest")
    assert commands[0][manifest_index + 1] == str(manifest)


def _panel(path: Path, names: list[str]) -> Path:
    rows = "".join(f"{name}\tpanel\n" for name in names)
    path.write_text("spec_name\tpanel\n" + rows)
    return path


def _prediction_row(spec_name: str, *, returned: bool) -> dict:
    return {
        "spec_name": spec_name,
        "lane": "dreams",
        "neutral_mass": 350.0,
        "runtime_seconds": 1.5,
        "attempts": 8,
        "truncated": False,
        "constraint_dead_ends": 5,
        "eos_terminated": 2,
        "max_length_terminated": 1,
        "validity": 0.25,
        "completed_validity": 0.5,
        "mass_validity": 0.125 if returned else 0.0,
        "uniqueness": 0.25 if returned else 0.0,
        "candidate_returned": returned,
        "exact_top1": False,
        "exact_top10": False,
        "tanimoto_top1": 0.4 if returned else 0.0,
        "tanimoto_top10": 0.5 if returned else 0.0,
        "formula_top1": False,
        "formula_top10": returned,
        "candidates": [],
    }


class _FakeProcess:
    def __init__(self, return_code: int) -> None:
        self.return_code = return_code

    def wait(self) -> int:
        return self.return_code


def _fake_popen(record, *, return_code=0, returned_names=None, drop=()):
    def popen(command, **kwargs):
        output = Path(command[command.index("--output-dir") + 1])
        manifest = Path(command[command.index("--spec-manifest") + 1])
        record.append({"command": list(command), "env": dict(kwargs.get("env", {}))})
        names = [
            line.split("\t")[0]
            for line in manifest.read_text().splitlines()[1:]
            if line.strip()
        ]
        output.mkdir(parents=True, exist_ok=True)
        rows = [
            _prediction_row(
                name,
                returned=returned_names is None or name in returned_names,
            )
            for name in names
            if name not in drop
        ]
        (output / "predictions.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
        (output / "metrics.json").write_text("{}\n")
        return _FakeProcess(return_code)

    return popen


class _Logger:
    def __init__(self) -> None:
        self.scalars = []
        self.texts = []

    def report_scalar(self, **kwargs):
        self.scalars.append(kwargs)

    def report_text(self, message):
        self.texts.append(message)


class _Task:
    id = "task-id"

    def __init__(self) -> None:
        self.logger = _Logger()

    def get_logger(self):
        return self.logger


def _config(tmp_path: Path, **evaluation) -> OmegaConf:
    base = {
        "enabled": True,
        "interval_steps": 1000,
        "metadata": str(tmp_path / "metadata.csv"),
        "fingerprints": str(tmp_path / "dreams_predictions.npz"),
        "fingerprint_key": "probs",
        "lane": "dreams",
        "candidates": 8,
        "diversity_dropout": 0.3,
        "temperature": 1.0,
        "ppm_tolerance": 10.0,
        "seed": 42,
    }
    base.update(evaluation)
    return OmegaConf.create(
        {
            "data": {"tokenizer_file": str(tmp_path / "tokenizer.json")},
            "evaluation": base,
            "output": {
                "root": str(tmp_path),
                "checkpoints": str(tmp_path / "checkpoints"),
                "checkpoint_interval": 1000,
            },
        }
    )


def _checkpoint(tmp_path: Path, step: int) -> Path:
    checkpoint = tmp_path / f"checkpoints/step={step}.ckpt"
    checkpoint.parent.mkdir(exist_ok=True)
    checkpoint.write_bytes(b"weights")
    return checkpoint


def test_unsharded_evaluation_keeps_the_single_process_command(
    tmp_path: Path, monkeypatch
) -> None:
    """The default must stay one process attached to the training task."""
    manifest = _panel(tmp_path / "panel.tsv", ["a", "b"])
    config = _config(tmp_path, spec_manifest=str(manifest), max_spectra=2)
    _checkpoint(tmp_path, 1000)
    commands = []
    monkeypatch.setattr(
        periodic_evaluation.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command),
    )
    monkeypatch.setattr(periodic_evaluation.torch.cuda, "is_available", lambda: False)

    callback = PeriodicMolecularEvaluation(
        config, project_root=tmp_path, clearml_task=_Task()
    )
    callback._run(1000)

    assert callback.shards == 1
    assert callback.selector is None
    assert "--clearml-task-id" in commands[0]
    assert commands[0][commands[0].index("--max-spectra") + 1] == "2"


def test_sharded_evaluation_requires_a_panel_manifest(tmp_path: Path) -> None:
    config = _config(tmp_path, shards=4)

    with pytest.raises(ValueError, match="needs evaluation.spec_manifest"):
        PeriodicMolecularEvaluation(config, project_root=tmp_path)


def test_sharded_evaluation_splits_merges_and_reports(
    tmp_path: Path, monkeypatch
) -> None:
    names = [f"spec{index}" for index in range(7)]
    manifest = _panel(tmp_path / "panel.tsv", names)
    config = _config(
        tmp_path, spec_manifest=str(manifest), max_spectra=7, shards=3
    )
    _checkpoint(tmp_path, 1000)
    launched = []
    monkeypatch.setattr(
        periodic_evaluation.subprocess,
        "Popen",
        _fake_popen(launched, returned_names={"spec0", "spec3"}),
    )
    monkeypatch.setattr(periodic_evaluation.torch.cuda, "is_available", lambda: False)
    task = _Task()

    PeriodicMolecularEvaluation(
        config, project_root=tmp_path, clearml_task=task
    )._run(1000)

    output = tmp_path / "periodic_molecular/step=1000"
    merged = [
        json.loads(line)
        for line in (output / "predictions.jsonl").read_text().splitlines()
    ]
    metrics = json.loads((output / "metrics.json").read_text())

    assert len(launched) == 3
    assert [row["spec_name"] for row in merged] == names
    assert metrics["rows"] == 7
    assert metrics["candidate_return_rate"] == pytest.approx(2 / 7)
    assert metrics["tanimoto_top1"] == pytest.approx(0.4)
    assert metrics["sharding"]["shards"] == 3
    for launch in launched:
        assert "--clearml-task-id" not in launch["command"]
        assert "--max-spectra" not in launch["command"]
        assert launch["env"]["OMP_NUM_THREADS"] == "1"
    candidate_return = [
        entry for entry in task.logger.scalars if entry["series"] == "Candidate return"
    ]
    assert candidate_return == [
        {
            "title": "Molecular screening",
            "series": "Candidate return",
            "value": pytest.approx(2 / 7),
            "iteration": 1000,
        }
    ]


def test_failed_shard_never_kills_training(tmp_path: Path, monkeypatch) -> None:
    manifest = _panel(tmp_path / "panel.tsv", ["a", "b", "c", "d"])
    config = _config(tmp_path, spec_manifest=str(manifest), shards=2)
    _checkpoint(tmp_path, 1000)
    monkeypatch.setattr(
        periodic_evaluation.subprocess, "Popen", _fake_popen([], return_code=1)
    )
    monkeypatch.setattr(periodic_evaluation.torch.cuda, "is_available", lambda: False)
    trainer = SimpleNamespace(is_global_zero=True, global_step=1000)

    PeriodicMolecularEvaluation(config, project_root=tmp_path).on_train_batch_end(
        trainer, None, None, None, 0
    )

    failure = json.loads(
        (tmp_path / "periodic_molecular/step=1000/failure.json").read_text()
    )
    assert "shards failed" in failure["error"]


def test_partial_panel_is_refused_rather_than_scored(
    tmp_path: Path, monkeypatch
) -> None:
    """A panel covering fewer spectra is not comparable with a full one, so it
    must never reach checkpoint selection."""
    manifest = _panel(tmp_path / "panel.tsv", ["a", "b", "c", "d"])
    config = _config(tmp_path, spec_manifest=str(manifest), shards=2)
    _checkpoint(tmp_path, 1000)
    monkeypatch.setattr(
        periodic_evaluation.subprocess, "Popen", _fake_popen([], drop=("c",))
    )
    monkeypatch.setattr(periodic_evaluation.torch.cuda, "is_available", lambda: False)
    trainer = SimpleNamespace(is_global_zero=True, global_step=1000)

    PeriodicMolecularEvaluation(config, project_root=tmp_path).on_train_batch_end(
        trainer, None, None, None, 0
    )

    failure = json.loads(
        (tmp_path / "periodic_molecular/step=1000/failure.json").read_text()
    )
    assert "3 of 4 panel spectra" in failure["error"]
    assert not (tmp_path / "periodic_molecular/step=1000/metrics.json").is_file()


def test_selection_records_the_best_checkpoint_and_stops_when_it_stalls(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        selection={
            "enabled": True,
            "metric": "candidate_return_rate",
            "patience": 2,
            "min_delta": 0.0,
        },
    )
    task = _Task()
    callback = PeriodicMolecularEvaluation(
        config, project_root=tmp_path, clearml_task=task
    )
    trajectory = {1000: 0.34, 2000: 0.28, 3000: 0.21}

    def write_metrics(step: int) -> None:
        output = tmp_path / f"periodic_molecular/step={step}"
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(
            json.dumps({"rows": 396, "candidate_return_rate": trajectory[step]})
        )

    callback._run = write_metrics
    trainer = SimpleNamespace(is_global_zero=True, global_step=0, should_stop=False)
    for step in trajectory:
        _checkpoint(tmp_path, step)
        trainer.global_step = step
        callback.on_train_batch_end(trainer, None, None, None, 0)

    state = json.loads((tmp_path / "selection/selection.json").read_text())
    link = tmp_path / "selection/best.ckpt"

    assert state["best_step"] == 1000
    assert state["patience"] == 2
    assert [entry["step"] for entry in state["history"]] == [1000, 2000, 3000]
    assert link.is_symlink()
    assert link.resolve() == (tmp_path / "checkpoints/step=1000.ckpt").resolve()
    assert trainer.should_stop is True
    assert any("Early stopping" in text for text in task.logger.texts)


def test_selection_failure_never_kills_training(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        selection={"enabled": True, "metric": "candidate_return_rate", "patience": 1},
    )
    callback = PeriodicMolecularEvaluation(config, project_root=tmp_path)

    def write_unusable_metrics(step: int) -> None:
        output = tmp_path / f"periodic_molecular/step={step}"
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(json.dumps({"rows": 0}))

    callback._run = write_unusable_metrics
    _checkpoint(tmp_path, 1000)
    trainer = SimpleNamespace(is_global_zero=True, global_step=1000, should_stop=False)

    callback.on_train_batch_end(trainer, None, None, None, 0)

    assert trainer.should_stop is False
    assert (tmp_path / "periodic_molecular/step=1000/failure.json").is_file()


def test_every_flag_the_callback_emits_is_a_flag_the_evaluator_accepts(
    tmp_path: Path,
) -> None:
    """The command is assembled as a string list, so a renamed evaluator flag
    would only surface as a failed evaluation inside a training run."""
    from scripts.evaluate_marlin_nplib1 import parse_args as evaluator_parse_args

    manifest = _panel(tmp_path / "panel.tsv", ["a", "b"])
    config = _config(
        tmp_path,
        spec_manifest=str(manifest),
        max_spectra=2,
        threshold=0.95,
        use_ema=False,
        sample_tokens=True,
        soft_fingerprint=True,
        mass_reachability_prune=True,
        forbid_isotope_tokens=True,
        restrict_organic_elements=True,
        per_spectrum_seconds=1800,
        shards=2,
    )
    callback = PeriodicMolecularEvaluation(
        config, project_root=tmp_path, clearml_task=_Task()
    )

    for command in (
        callback._command(
            1000,
            tmp_path / "step=1000.ckpt",
            tmp_path / "out",
            spec_manifest=manifest,
            max_spectra=2,
        ),
        callback._command(
            1000,
            tmp_path / "step=1000.ckpt",
            tmp_path / "out",
            spec_manifest=manifest,
            attach_clearml=False,
        ),
    ):
        import sys

        assert command[3].endswith("evaluate_marlin_nplib1.py")
        argv = [str(part) for part in command[4:]]
        original = sys.argv
        sys.argv = ["evaluate_marlin_nplib1.py", *argv]
        try:
            parsed = evaluator_parse_args()
        finally:
            sys.argv = original
        assert parsed.mass_reachability_prune
        assert parsed.forbid_isotope_tokens
        assert parsed.restrict_organic_elements
        assert parsed.per_spectrum_seconds == 1800
        assert parsed.no_ema
        assert parsed.sample_tokens
        assert parsed.soft_fingerprint
