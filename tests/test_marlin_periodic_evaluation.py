from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from marlin.periodic_evaluation import PeriodicMolecularEvaluation


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
