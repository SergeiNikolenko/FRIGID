"""Fail-soft periodic molecular evaluation for MARLIN training."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path

import lightning as L
import torch

from marlin.benchmark_selection import (
    load_spec_manifest,
    write_interleaved_shard_manifests,
)
from marlin.checkpoint_selection import CheckpointSelector
from marlin.evaluation import (
    MOLECULAR_SCALAR_SERIES,
    aggregate_prediction_metrics,
)


# One BLAS thread per shard. Sixteen torch processes each defaulting to every
# core drove the load average to 86 and ran slower than serial.
_SHARD_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


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
        self.shards = self._resolve_shards()
        self.selector = self._build_selector()

    def _resolve_shards(self) -> int:
        evaluation = self.config.get("evaluation")
        if not evaluation:
            return 1
        shards = int(evaluation.get("shards", 1) or 1)
        if shards < 1:
            raise ValueError("evaluation.shards must be at least 1")
        if shards > 1 and not evaluation.get("spec_manifest"):
            raise ValueError(
                "sharded evaluation needs evaluation.spec_manifest: shards are "
                "cut from the panel manifest, not from the metadata order"
            )
        return shards

    def _build_selector(self) -> CheckpointSelector | None:
        evaluation = self.config.get("evaluation")
        selection = evaluation.get("selection") if evaluation else None
        if not selection or not bool(selection.get("enabled", False)):
            return None
        return CheckpointSelector(
            metric=str(selection.get("metric", "candidate_return_rate")),
            patience=int(selection.get("patience", 0) or 0),
            min_delta=float(selection.get("min_delta", 0.0) or 0.0),
        )

    def _checkpoint(self, step: int) -> Path:
        return Path(self.config.output.checkpoints) / f"step={step}.ckpt"

    def _output(self, step: int) -> Path:
        return Path(self.config.output.root) / "periodic_molecular" / f"step={step}"

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

    def _command(
        self,
        step: int,
        checkpoint: Path,
        output: Path,
        *,
        spec_manifest: Path | str | None = None,
        max_spectra: object = None,
        attach_clearml: bool = True,
    ) -> list[str]:
        evaluation = self.config.evaluation
        command = [
            sys.executable,
            "-X",
            "faulthandler",
            str(self.project_root / "scripts" / "evaluate_marlin_nplib1.py"),
            "--checkpoint", str(checkpoint),
            "--tokenizer", str(self.config.data.tokenizer_file),
            "--metadata", str(evaluation.metadata),
            "--fingerprints", str(evaluation.fingerprints),
            "--fingerprint-key", str(evaluation.fingerprint_key),
            "--lane", str(evaluation.lane),
            "--output-dir", str(output),
            "--candidates", str(evaluation.candidates),
            "--diversity-dropout", str(evaluation.diversity_dropout),
            "--temperature", str(evaluation.temperature),
            "--generation-mode", "block",
            "--ppm-tolerance", str(evaluation.ppm_tolerance),
            "--seed", str(evaluation.seed),
            "--clearml-iteration", str(step),
            "--evaluation-profile", "screening",
        ]
        if max_spectra is not None and str(max_spectra):
            command.extend(["--max-spectra", str(max_spectra)])
        if spec_manifest is not None and str(spec_manifest):
            command.extend(["--spec-manifest", str(spec_manifest)])
        threshold = evaluation.get("threshold")
        if threshold is not None and str(threshold):
            command.extend(["--threshold", str(threshold)])
        if not bool(evaluation.get("use_ema", True)):
            command.append("--no-ema")
        if bool(evaluation.get("sample_tokens", False)):
            command.append("--sample-tokens")
        if bool(evaluation.get("soft_fingerprint", False)):
            command.append("--soft-fingerprint")
        if bool(evaluation.get("mass_reachability_prune", False)):
            command.append("--mass-reachability-prune")
        if bool(evaluation.get("forbid_isotope_tokens", False)):
            command.append("--forbid-isotope-tokens")
        if bool(evaluation.get("restrict_organic_elements", False)):
            command.append("--restrict-organic-elements")
        per_spectrum_seconds = evaluation.get("per_spectrum_seconds")
        if per_spectrum_seconds is not None and str(per_spectrum_seconds):
            command.extend(["--per-spectrum-seconds", str(per_spectrum_seconds)])
        if str(self.config.get("architecture", "marlin")) == "expanding":
            command.extend(
                [
                    "--architecture",
                    "expanding",
                    "--expanding-steps",
                    str(evaluation.get("expanding_steps", 32)),
                    "--disable-grammar-mask",
                ]
            )
        if attach_clearml and self.clearml_task is not None:
            command.extend(["--clearml-task-id", self.clearml_task.id])
        return command

    def _environment(self, *, sharded: bool) -> dict[str, str]:
        environment = dict(os.environ)
        environment["PYTHONFAULTHANDLER"] = "1"
        if sharded:
            environment.update(_SHARD_THREAD_ENVIRONMENT)
        return environment

    def _run_single(self, step: int, checkpoint: Path, output: Path) -> None:
        evaluation = self.config.evaluation
        command = self._command(
            step,
            checkpoint,
            output,
            spec_manifest=evaluation.get("spec_manifest"),
            max_spectra=evaluation.get("max_spectra"),
            attach_clearml=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        subprocess.run(
            command,
            cwd=self.project_root,
            check=True,
            env=self._environment(sharded=False),
        )

    def _run_sharded(self, step: int, checkpoint: Path, output: Path) -> None:
        """Evaluate one panel as ``shards`` concurrent single-core processes.

        The constrained decoder is CPU bound, so a panel splits across cores at
        close to linear speedup: the 803-spectrum split at 8 candidates takes
        about 1.1 h on 16 shards against about 17 h in one process. That is what
        makes a 396-spectrum validation panel fit inside a 10,000-step interval.

        Each shard costs about one core and roughly 2 GB of device memory
        (measured on twelve concurrent evaluators holding 22 GB in total), so 16
        shards sit beside a training process on one A100 rather than replacing it.
        """
        evaluation = self.config.evaluation
        manifest = Path(str(evaluation.spec_manifest))
        names = load_spec_manifest(manifest)
        if self.shards > len(names):
            raise ValueError(
                f"evaluation.shards={self.shards} exceeds the {len(names)}-spectrum "
                f"panel {manifest}"
            )
        output.mkdir(parents=True, exist_ok=True)
        shard_manifests = write_interleaved_shard_manifests(
            manifest, output / "shards", self.shards
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        environment = self._environment(sharded=True)
        running = []
        for index, shard_manifest in enumerate(shard_manifests):
            shard_output = output / f"shard{index:02d}"
            if (shard_output / "metrics.json").is_file():
                continue
            command = self._command(
                step,
                checkpoint,
                shard_output,
                spec_manifest=shard_manifest,
                # max_spectra is deliberately dropped: the shard manifest already
                # fixes the rows, and the evaluator refuses a --max-spectra that
                # does not equal its manifest length.
                max_spectra=None,
                # Every shard reports into one merged panel, so attaching each of
                # them to the training task would publish `shards` conflicting
                # scalar points at the same iteration.
                attach_clearml=False,
            )
            log = (output / f"shard{index:02d}.log").open("w")
            running.append(
                (
                    index,
                    subprocess.Popen(
                        command,
                        cwd=self.project_root,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    ),
                    log,
                )
            )
        failures = []
        for index, process, log in running:
            return_code = process.wait()
            log.close()
            if return_code != 0:
                failures.append((index, return_code))
        if failures:
            raise RuntimeError(
                "sharded evaluation shards failed: "
                + ", ".join(f"shard{index:02d} rc={code}" for index, code in failures)
                + f"; logs under {output}"
            )
        metrics = self._merge_shards(output, names)
        self._report_metrics(step, metrics)

    def _merge_shards(self, output: Path, names: list[str]) -> dict:
        order = {name: position for position, name in enumerate(names)}
        rows: list[dict] = []
        seen: set[str] = set()
        for index in range(self.shards):
            predictions = output / f"shard{index:02d}" / "predictions.jsonl"
            if not predictions.is_file():
                raise FileNotFoundError(f"shard produced no predictions: {predictions}")
            for line in predictions.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                spec_name = str(row["spec_name"])
                if spec_name in seen:
                    raise ValueError(
                        f"duplicate spectrum across shards: {spec_name}"
                    )
                if spec_name not in order:
                    raise ValueError(
                        f"shard returned a spectrum outside the panel: {spec_name}"
                    )
                seen.add(spec_name)
                rows.append(row)
        if len(rows) != len(names):
            # A partial panel is not comparable with a full one, so it must not
            # reach checkpoint selection.
            raise ValueError(
                f"merged shards cover {len(rows)} of {len(names)} panel spectra"
            )
        rows.sort(key=lambda row: order[str(row["spec_name"])])
        (output / "predictions.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
        metrics = aggregate_prediction_metrics(rows)
        metrics["sharding"] = {
            "shards": self.shards,
            "merged_from": [f"shard{index:02d}" for index in range(self.shards)],
            "omitted_metrics": ["internal_diversity"],
        }
        (output / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n"
        )
        return metrics

    def _report_metrics(self, step: int, metrics: dict) -> None:
        """Publish merged panel metrics under the same series as one process.

        The single-process path reports these from the evaluator itself; a sharded
        panel has no single evaluator to do it, and the periodic evaluation always
        runs the screening profile, whose title is "Molecular screening".
        """
        if self.clearml_task is None:
            return
        logger = self.clearml_task.get_logger()
        for series, key in MOLECULAR_SCALAR_SERIES.items():
            if key not in metrics:
                continue
            value = float(metrics[key])
            if not math.isfinite(value):
                continue
            logger.report_scalar(
                title="Molecular screening",
                series=series,
                value=value,
                iteration=step,
            )

    def _run(self, step: int) -> None:
        checkpoint = self._checkpoint(step)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not available for evaluation: {checkpoint}")
        output = self._output(step)
        if (output / "metrics.json").is_file():
            return
        if self.shards > 1:
            self._run_sharded(step, checkpoint, output)
        else:
            self._run_single(step, checkpoint, output)

    def _select(self, trainer, step: int) -> None:
        """Record the best checkpoint and stop the run when it stalls.

        Selection history is rebuilt from this process only: a resumed run
        starts its patience counter again, which is recorded in
        ``selection/selection.json`` rather than inferred.
        """
        if self.selector is None:
            return
        metrics_path = self._output(step) / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        decision = self.selector.update(step, metrics)
        directory = Path(self.config.output.root) / "selection"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "selection.json").write_text(
            json.dumps(self.selector.state(), indent=2, sort_keys=True) + "\n"
        )
        best_checkpoint = self._checkpoint(decision.best_step)
        link = directory / "best.ckpt"
        if best_checkpoint.is_file():
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(os.path.relpath(best_checkpoint, directory))
        print(
            f"Checkpoint selection at step {step}: {decision.metric}="
            f"{decision.value:.4f}, best={decision.best_value:.4f} at step "
            f"{decision.best_step}, evaluations since best "
            f"{decision.evaluations_since_best}",
            flush=True,
        )
        if self.clearml_task is not None:
            logger = self.clearml_task.get_logger()
            logger.report_scalar(
                title="Checkpoint selection",
                series=f"{decision.metric} (selection)",
                value=decision.value,
                iteration=step,
            )
            logger.report_scalar(
                title="Checkpoint selection",
                series="best step",
                value=float(decision.best_step),
                iteration=step,
            )
            logger.report_scalar(
                title="Checkpoint selection",
                series="evaluations since best",
                value=float(decision.evaluations_since_best),
                iteration=step,
            )
        if decision.should_stop:
            print(f"Early stopping MARLIN adaptation: {decision.reason}", flush=True)
            if self.clearml_task is not None:
                self.clearml_task.get_logger().report_text(
                    f"Early stopping at step {step}: {decision.reason}"
                )
            trainer.should_stop = True

    def _maybe_run(self, trainer) -> None:
        evaluation = self.config.get("evaluation")
        if not trainer.is_global_zero or not evaluation or not bool(evaluation.enabled):
            return
        step = int(trainer.global_step)
        interval = int(evaluation.interval_steps)
        checkpoint_step = step - step % interval
        if checkpoint_step <= 0 or checkpoint_step in self._completed_steps:
            return
        # A resumed run initially reports the source global step before its
        # first optimizer update, but that checkpoint belongs to another root.
        if not self._checkpoint(checkpoint_step).is_file():
            return
        try:
            self._run(checkpoint_step)
            self._select(trainer, checkpoint_step)
        except Exception as error:  # evaluation must never destroy a training run
            self._report_failure(checkpoint_step, error)
        finally:
            self._completed_steps.add(checkpoint_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        del pl_module, outputs, batch, batch_idx
        self._maybe_run(trainer)

    def on_train_end(self, trainer, pl_module) -> None:
        del pl_module
        # ModelCheckpoint may write the final checkpoint after this callback's
        # last on_train_batch_end hook. Retry once after all training batches.
        self._maybe_run(trainer)
