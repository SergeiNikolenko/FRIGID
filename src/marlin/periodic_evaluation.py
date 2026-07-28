"""Fail-soft periodic molecular evaluation for MARLIN training."""

from __future__ import annotations

import json
import subprocess
import sys
import traceback
from pathlib import Path

import lightning as L
import torch


class PeriodicMolecularEvaluation(L.Callback):
    """Run the canonical held-out evaluator after selected checkpoints."""

    def __init__(self, config, *, project_root: Path, clearml_task=None) -> None:
        super().__init__()
        self.config = config
        self.project_root = project_root
        self.clearml_task = clearml_task
        self._completed_steps: set[int] = set()
        if (
            bool(config.evaluation.enabled)
            and int(config.evaluation.interval_steps)
            != int(config.output.checkpoint_interval)
        ):
            raise ValueError(
                "evaluation.interval_steps must equal output.checkpoint_interval"
            )

    def _checkpoint(self, step: int) -> Path:
        return Path(self.config.output.checkpoints) / f"step={step}.ckpt"

    def _report_failure(self, step: int, error: BaseException) -> None:
        message = "".join(traceback.format_exception(error)).strip()
        print(f"Periodic molecular evaluation failed at step {step}:\n{message}", flush=True)
        output = Path(self.config.output.root) / "periodic_molecular" / f"step={step}"
        output.mkdir(parents=True, exist_ok=True)
        (output / "failure.json").write_text(
            json.dumps({"step": step, "error": message}, indent=2) + "\n"
        )
        if self.clearml_task is not None:
            self.clearml_task.get_logger().report_text(
                f"Periodic molecular evaluation failed at step {step}: {error}"
            )

    def _run(self, step: int) -> None:
        evaluation = self.config.evaluation
        checkpoint = self._checkpoint(step)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not available for evaluation: {checkpoint}")
        output = Path(self.config.output.root) / "periodic_molecular" / f"step={step}"
        if (output / "metrics.json").is_file():
            return
        command = [
            sys.executable,
            str(self.project_root / "scripts" / "evaluate_marlin_nplib1.py"),
            "--checkpoint", str(checkpoint),
            "--tokenizer", str(self.config.data.tokenizer_file),
            "--metadata", str(evaluation.metadata),
            "--fingerprints", str(evaluation.fingerprints),
            "--fingerprint-key", str(evaluation.fingerprint_key),
            "--lane", str(evaluation.lane),
            "--output-dir", str(output),
            "--candidates", str(evaluation.candidates),
            "--max-spectra", str(evaluation.max_spectra),
            "--diversity-dropout", str(evaluation.diversity_dropout),
            "--temperature", str(evaluation.temperature),
            "--generation-mode", "block",
            "--ppm-tolerance", str(evaluation.ppm_tolerance),
            "--seed", str(evaluation.seed),
            "--clearml-iteration", str(step),
        ]
        if self.clearml_task is not None:
            command.extend(["--clearml-task-id", self.clearml_task.id])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        subprocess.run(command, cwd=self.project_root, check=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        del pl_module, outputs, batch, batch_idx
        evaluation = self.config.get("evaluation")
        if not trainer.is_global_zero or not evaluation or not bool(evaluation.enabled):
            return
        step = int(trainer.global_step)
        interval = int(evaluation.interval_steps)
        if step <= 0 or step % interval or step in self._completed_steps:
            return
        # A resumed run initially reports the source global step before its
        # first optimizer update, but that checkpoint belongs to another root.
        if not self._checkpoint(step).is_file():
            return
        try:
            self._run(step)
            self._completed_steps.add(step)
        except Exception as error:  # evaluation must never destroy a training run
            self._report_failure(step, error)
