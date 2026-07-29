#!/usr/bin/env python3
# ruff: noqa: E402
"""Train conditional SAFE EFlow teachers and EFM students."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import datasets
import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from marlin.dataset import (
    streaming_loader_workers,
    verify_filtered_prefix_cache,
    verify_snapshot_manifest,
)
from marlin.expanding import ExpandingFlowConfig
from marlin.expanding_checkpoint import expanding_model_from_checkpoint
from marlin.expanding_training import ExpandingMarlinLightningModule
from marlin.model import MarlinDecoderConfig
from marlin.periodic_evaluation import PeriodicMolecularEvaluation
from marlin.tokenizer import load_safe_tokenizer, validate_safe_tokenizer
from marlin.training import MarlinCollator, MarlinTrainingFilter
from marlin.warm_start import load_frigid_decoder, sha256_file


def git_state() -> tuple[str | None, list[str]]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
        ).splitlines()
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, []


def package_versions(names: list[str]) -> dict[str, str | None]:
    resolved = {}
    for name in names:
        try:
            resolved[name] = version(name)
        except PackageNotFoundError:
            resolved[name] = None
    return resolved


def write_run_manifest(
    config: DictConfig,
    *,
    tokenizer_sha256: str,
    snapshot_manifest_sha256: str,
) -> dict:
    commit, dirty = git_state()
    if dirty and not bool(config.get("allow_dirty_worktree", False)):
        raise RuntimeError(
            f"refusing expanding training from dirty git state: {dirty}"
        )
    manifest = {
        "schema_version": 1,
        "kind": "conditional MARLIN Expanding Flow training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "clean_room_reproduction": False,
        "paper_training_recipe": False,
        "training_variant": f"EFM-inspired MARLIN {config.stage}",
        "architecture_source": "arXiv:2607.21585v1",
        "target_task_source": "arXiv:2607.04774",
        "git_commit": commit,
        "dirty_worktree": dirty,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "config": OmegaConf.to_container(config, resolve=True),
        "inputs": {
            "tokenizer_sha256": tokenizer_sha256,
            "training_snapshot_manifest_sha256": snapshot_manifest_sha256,
            "frigid_warm_start_sha256": (
                str(config.frigid_warm_start_sha256)
                if config.get("frigid_warm_start_checkpoint")
                else None
            ),
            "teacher_checkpoint": (
                str(config.teacher_checkpoint)
                if config.get("teacher_checkpoint")
                else None
            ),
            "teacher_checkpoint_sha256": (
                sha256_file(config.teacher_checkpoint)
                if config.get("teacher_checkpoint")
                else None
            ),
        },
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "packages": package_versions(
                ["torch", "lightning", "datasets", "rdkit", "clearml"]
            ),
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
    }
    path = Path(config.output.root) / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def initialize_clearml(config: DictConfig):
    if not bool(config.tracking.clearml.enabled):
        return None
    from clearml import Task

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    stage = str(config.stage)
    tags = list(config.tracking.clearml.tags)
    tags.extend([stage, "variable-length", "mass-conditioned"])
    task = Task.init(
        project_name=str(config.tracking.clearml.project_name),
        task_name=f"{config.tracking.clearml.task_name}-{job_id}",
        tags=tags,
        reuse_last_task_id=False,
        output_uri=False,
        auto_connect_streams=False,
        auto_connect_frameworks={"pytorch": True, "tensorboard": True},
        auto_resource_monitoring={
            "report_frequency_sec": 5.0,
            "first_report_sec": 5.0,
            "wait_for_first_iteration_to_start_sec": 5.0,
            "max_wait_for_first_iteration_to_start_sec": 5.0,
        },
    )
    task.connect(
        OmegaConf.to_container(config, resolve=True), name="resolved_config"
    )
    commit, _ = git_state()
    path = Path(config.output.root) / "clearml_task.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "task_id": task.id,
                "task_name": task.name,
                "project_name": task.get_project_name(),
                "web_url": task.get_output_log_web_page(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "git_commit": commit,
                "stage": stage,
            },
            indent=2,
        )
        + "\n"
    )
    return task


def build_loader(
    config: DictConfig,
    tokenizer,
    decoder_config: MarlinDecoderConfig,
    *,
    local_shards: list[str],
    snapshot_manifest_sha256: str,
    tokenizer_sha256: str,
):
    source = datasets.load_dataset(
        "parquet",
        data_files={"train": local_shards},
        split="train",
        streaming=True,
        cache_dir=config.data.hf_cache_dir,
    )
    training_filter = MarlinTrainingFilter(
        tokenizer,
        decoder_config.max_length,
        config.data.exclude_inchikeys,
    )
    cache_manifest = Path(config.data.filtered_prefix_cache_manifest)
    uses_filtered_prefix = cache_manifest.is_file()
    if uses_filtered_prefix:
        cache_shard, raw_rows_consumed = verify_filtered_prefix_cache(
            cache_manifest,
            source_manifest_sha256=snapshot_manifest_sha256,
            tokenizer_sha256=tokenizer_sha256,
            exclusion_sha256=sha256_file(config.data.exclude_inchikeys),
            max_length=decoder_config.max_length,
            minimum_rows=int(config.data.shuffle_buffer),
        )
        cached = datasets.load_dataset(
            "parquet",
            data_files={"train": [cache_shard]},
            split="train",
            streaming=True,
            cache_dir=config.data.hf_cache_dir,
        )
        tail = source.skip(raw_rows_consumed).filter(training_filter)
        dataset = datasets.concatenate_datasets([cached, tail])
    else:
        dataset = source.filter(training_filter)
    dataset = dataset.shuffle(
        seed=int(config.seed), buffer_size=int(config.data.shuffle_buffer)
    )
    collator = MarlinCollator(
        tokenizer,
        max_length=decoder_config.max_length,
        fingerprint_bits=decoder_config.fingerprint_bits,
        exclude_inchikeys=config.data.exclude_inchikeys,
    )
    workers = streaming_loader_workers(
        int(config.loader.num_workers),
        uses_filtered_prefix=uses_filtered_prefix,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(config.loader.batch_size),
        num_workers=workers,
        pin_memory=True,
        collate_fn=collator,
    )


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="expanding_marlin_nplib1",
)
def main(config: DictConfig) -> None:
    stage = str(config.stage)
    if stage not in {"eflow", "efm"}:
        raise ValueError("stage must be eflow or efm")
    if stage == "eflow":
        sources = [
            name
            for name, value in (
                (
                    "frigid_warm_start_checkpoint",
                    config.get("frigid_warm_start_checkpoint"),
                ),
                ("resume_checkpoint", config.get("resume_checkpoint")),
            )
            if value
        ]
        if len(sources) != 1:
            raise ValueError(
                "EFlow requires exactly one initialization source; got "
                + (", ".join(sources) or "none")
            )
    if stage == "efm" and not config.get("teacher_checkpoint"):
        raise ValueError("EFM requires teacher_checkpoint")
    if stage == "efm" and config.get("frigid_warm_start_checkpoint"):
        raise ValueError(
            "EFM initializes from its EFlow teacher, not directly from FRIGID"
        )

    L.seed_everything(int(config.seed), workers=True)
    torch.set_float32_matmul_precision("high")
    tokenizer_sha256 = sha256_file(config.data.tokenizer_file)
    if tokenizer_sha256 != str(config.data.tokenizer_sha256):
        raise ValueError("SAFE tokenizer hash mismatch")
    tokenizer = load_safe_tokenizer(config.data.tokenizer_file)
    decoder_config = MarlinDecoderConfig(
        **OmegaConf.to_container(config.model, resolve=True)
    )
    flow_config = ExpandingFlowConfig(
        **OmegaConf.to_container(config.flow, resolve=True)
    )
    special_ids = validate_safe_tokenizer(
        tokenizer,
        expected_vocab_size=decoder_config.vocab_size,
        expected_special_token_ids=OmegaConf.to_container(
            config.data.special_token_ids, resolve=True
        ),
    )
    if special_ids["eos"] != decoder_config.eos_token_id:
        raise ValueError("model EOS token ID does not match tokenizer")
    length_audit = json.loads(
        Path(config.data.length_audit_manifest).read_text()
    )
    if (
        sha256_file(config.data.length_audit_manifest)
        != str(config.data.length_audit_sha256)
    ):
        raise ValueError("training length audit manifest hash mismatch")
    if length_audit["dataset_revision"] != str(config.data.revision):
        raise ValueError("training length audit dataset revision mismatch")
    if length_audit["maximum_allowed_length"] != decoder_config.max_length:
        raise ValueError("training length audit decoder context mismatch")
    if not length_audit.get("strict_safe_decode", False):
        raise ValueError("training audit did not use strict SAFE decoding")
    if length_audit.get("exclusion_sha256") != sha256_file(
        config.data.exclude_inchikeys
    ):
        raise ValueError("training audit exclusion hash mismatch")
    local_shards, snapshot_manifest_sha256 = verify_snapshot_manifest(
        config.data.snapshot_manifest,
        expected_dataset=str(config.data.dataset),
        expected_revision=str(config.data.revision),
        expected_file_list_sha256=str(config.data.snapshot_file_list_sha256),
        verify_hashes=bool(config.data.get("verify_snapshot_hashes", True)),
    )
    module = ExpandingMarlinLightningModule(
        decoder_config,
        flow_config,
        stage=stage,
        learning_rate=float(config.optim.learning_rate),
        weight_decay=float(config.optim.weight_decay),
        warmup_steps=int(config.optim.warmup_steps),
        noise_probability=float(config.training.noise_probability),
        noise_min_fraction=float(config.training.noise_min_fraction),
        noise_max_fraction=float(config.training.noise_max_fraction),
        ema_decay=float(config.training.ema_decay),
        metric_interval=int(config.training.metric_interval),
    )
    if stage == "eflow" and config.get("frigid_warm_start_checkpoint"):
        report = load_frigid_decoder(
            module.model.backbone,
            config.frigid_warm_start_checkpoint,
            expected_sha256=str(config.frigid_warm_start_sha256),
        )
        report.update(
            {
                "architecture": "conditional EFlow",
                "randomly_initialized_components": [
                    "source time embedder",
                    "target time embedder",
                    "per-layer time modulation",
                    "insertion head",
                ],
            }
        )
        module.reset_ema()
        path = Path(config.output.root) / "warm_start.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    if stage == "efm":
        teacher, teacher_stage = expanding_model_from_checkpoint(
            config.teacher_checkpoint,
            device="cpu",
            use_ema=True,
        )
        if teacher_stage != "eflow":
            raise ValueError("EFM teacher checkpoint must be an EFlow")
        if (
            teacher.decoder_config != decoder_config
            or teacher.flow_config != flow_config
        ):
            raise ValueError("EFM config must exactly match its EFlow teacher")
        module.model.load_state_dict(teacher.state_dict(), strict=True)
        module.set_teacher(teacher)
        module.reset_ema()

    write_run_manifest(
        config,
        tokenizer_sha256=tokenizer_sha256,
        snapshot_manifest_sha256=snapshot_manifest_sha256,
    )
    loader = build_loader(
        config,
        tokenizer,
        decoder_config,
        local_shards=local_shards,
        snapshot_manifest_sha256=snapshot_manifest_sha256,
        tokenizer_sha256=tokenizer_sha256,
    )
    clearml_task = initialize_clearml(config)
    checkpoint = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=config.output.checkpoints,
        filename="{step}",
        every_n_train_steps=int(config.output.checkpoint_interval),
        save_top_k=-1,
        save_last=True,
    )
    molecular_evaluation = PeriodicMolecularEvaluation(
        config,
        project_root=PROJECT_ROOT,
        clearml_task=clearml_task,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=int(config.trainer.devices),
        strategy=(
            "ddp" if int(config.trainer.devices) > 1 else "auto"
        ),
        precision=str(config.trainer.precision),
        max_steps=int(config.trainer.max_steps),
        accumulate_grad_batches=int(config.trainer.accumulate_grad_batches),
        gradient_clip_val=float(config.trainer.gradient_clip_val),
        log_every_n_steps=int(config.trainer.log_every_n_steps),
        callbacks=[checkpoint, molecular_evaluation],
        default_root_dir=config.output.root,
    )
    try:
        trainer.fit(
            module,
            loader,
            ckpt_path=config.get("resume_checkpoint"),
        )
    finally:
        if clearml_task is not None:
            clearml_task.close()


if __name__ == "__main__":
    main()
