#!/usr/bin/env python3
# ruff: noqa: E402
"""Train the clean-room MARLIN block-diffusion decoder."""

from __future__ import annotations

import os
import sys
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import datasets
import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from marlin.model import MarlinDecoderConfig
from marlin.tokenizer import load_safe_tokenizer, validate_safe_tokenizer
from marlin.training import (
    MarlinCollator,
    MarlinLightningModule,
    SafeLengthFilter,
)
from marlin.warm_start import load_frigid_decoder, sha256_file


def git_state() -> tuple[str | None, list[str]]:
    """Return the checked-out commit and any uncommitted paths."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, []


def package_versions(names: list[str]) -> dict[str, str | None]:
    """Resolve a compact environment inventory without invoking pip."""
    resolved = {}
    for name in names:
        try:
            resolved[name] = version(name)
        except PackageNotFoundError:
            resolved[name] = None
    return resolved


def write_run_manifest(config: DictConfig, tokenizer_sha256: str) -> dict:
    """Persist the immutable training inputs and execution environment."""
    commit, dirty = git_state()
    if dirty:
        raise RuntimeError(f"refusing canonical training from dirty git state: {dirty}")
    manifest = {
        "schema_version": 1,
        "kind": "MARLIN clean-room decoder training",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "clean_room_reproduction": True,
        "author_code_available_at_start": False,
        "git_commit": commit,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "config": OmegaConf.to_container(config, resolve=True),
        "inputs": {
            "tokenizer_sha256": tokenizer_sha256,
            "warm_start_sha256": str(config.warm_start_sha256),
            "nplib1_test_exclusions_sha256": sha256_file(
                config.data.exclude_inchikeys
            ),
            "training_length_audit_sha256": sha256_file(
                config.data.length_audit_manifest
            ),
            "dataset": str(config.data.dataset),
            "dataset_revision": str(config.data.revision),
        },
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(
                [
                    "torch",
                    "lightning",
                    "datasets",
                    "transformers",
                    "rdkit",
                    "safe-mol",
                    "clearml",
                ]
            ),
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "inferred_parameters": [
            "training corpus and split",
            "dataset revision and shuffle buffer",
            "max_steps=100000",
            "maximum sequence length=256",
            "exclude complete SAFE targets longer than 256 tokens",
            "FFN width, dropout, gradient clipping, and weight decay",
            "absence of a learning-rate schedule and warmup",
            "64 mass Fourier frequencies from 1e-3 to 1.0",
            "theoretical isotope-envelope calculation",
            "data-loader and GPU execution settings",
        ],
    }
    output = Path(config.output.root) / "run_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def initialize_clearml(config: DictConfig):
    """Create the experiment record before Lightning initializes its logger."""
    if not config.tracking.clearml.enabled:
        return None
    from clearml import Task

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    task = Task.init(
        project_name=config.tracking.clearml.project_name,
        task_name=f"{config.tracking.clearml.task_name}-{job_id}",
        tags=list(config.tracking.clearml.tags),
        reuse_last_task_id=False,
        output_uri=False,
        auto_connect_frameworks={"pytorch": True, "tensorboard": True},
    )
    resolved_config = OmegaConf.to_container(config, resolve=True)
    task.connect(resolved_config, name="resolved_config")
    commit, _ = git_state()
    tracking_path = Path(config.output.root) / "clearml_task.json"
    tracking_path.parent.mkdir(parents=True, exist_ok=True)
    tracking_path.write_text(
        json.dumps(
            {
                "task_id": task.id,
                "task_name": task.name,
                "project_name": task.get_project_name(),
                "web_url": task.get_output_log_web_page(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "git_commit": commit,
            },
            indent=2,
        )
        + "\n"
    )
    return task


@hydra.main(version_base=None, config_path="../configs", config_name="marlin_nplib1")
def main(config: DictConfig) -> None:
    L.seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    tokenizer_sha256 = sha256_file(config.data.tokenizer_file)
    if tokenizer_sha256 != config.data.tokenizer_sha256:
        raise ValueError(
            f"SAFE tokenizer SHA-256 is {tokenizer_sha256}; "
            f"expected {config.data.tokenizer_sha256}"
        )
    tokenizer = load_safe_tokenizer(config.data.tokenizer_file)
    decoder_config = MarlinDecoderConfig(
        **OmegaConf.to_container(config.model, resolve=True)
    )
    special_token_ids = validate_safe_tokenizer(
        tokenizer,
        expected_vocab_size=decoder_config.vocab_size,
        expected_special_token_ids=OmegaConf.to_container(
            config.data.special_token_ids, resolve=True
        ),
    )
    if special_token_ids["mask"] != decoder_config.mask_token_id:
        raise ValueError("model MASK token ID does not match the SAFE tokenizer")
    if special_token_ids["pad"] != decoder_config.pad_token_id:
        raise ValueError("model PAD token ID does not match the SAFE tokenizer")
    write_run_manifest(config, tokenizer_sha256)
    module = MarlinLightningModule(
        decoder_config,
        learning_rate=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
        noise_probability=config.training.noise_probability,
        noise_min_fraction=config.training.noise_min_fraction,
        noise_max_fraction=config.training.noise_max_fraction,
        ema_decay=config.training.ema_decay,
    )
    if config.get("warm_start_checkpoint"):
        report = load_frigid_decoder(
            module.decoder,
            config.warm_start_checkpoint,
            expected_sha256=config.get("warm_start_sha256"),
        )
        report["tokenizer"] = str(config.data.tokenizer_file)
        report["tokenizer_sha256"] = tokenizer_sha256
        report["special_token_ids"] = special_token_ids
        report["dataset"] = str(config.data.dataset)
        report["dataset_revision"] = str(config.data.revision)
        module.reset_ema()
        report_path = Path(config.output.root) / "warm_start.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    length_audit = json.loads(Path(config.data.length_audit_manifest).read_text())
    if sha256_file(config.data.length_audit_manifest) != config.data.length_audit_sha256:
        raise ValueError("training length audit manifest hash mismatch")
    if length_audit["dataset_revision"] != str(config.data.revision):
        raise ValueError("training length audit dataset revision mismatch")
    if length_audit["maximum_allowed_length"] != decoder_config.max_length:
        raise ValueError("training length audit decoder context mismatch")
    dataset = datasets.load_dataset(
        config.data.dataset,
        revision=config.data.revision,
        split="train",
        streaming=True,
        cache_dir=config.data.hf_cache_dir,
    ).filter(
        SafeLengthFilter(tokenizer, decoder_config.max_length),
    )
    dataset = dataset.shuffle(seed=config.seed, buffer_size=config.data.shuffle_buffer)
    collator = MarlinCollator(
        tokenizer,
        max_length=decoder_config.max_length,
        fingerprint_bits=decoder_config.fingerprint_bits,
        exclude_inchikeys=config.data.exclude_inchikeys,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.loader.batch_size,
        num_workers=config.loader.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    clearml_task = initialize_clearml(config)
    checkpoint = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=config.output.checkpoints,
        filename="{step}",
        every_n_train_steps=config.output.checkpoint_interval,
        save_top_k=-1,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=config.trainer.devices,
        strategy="ddp" if config.trainer.devices > 1 else "auto",
        precision=config.trainer.precision,
        max_steps=config.trainer.max_steps,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        gradient_clip_val=1.0,
        log_every_n_steps=config.trainer.log_every_n_steps,
        callbacks=[checkpoint],
        default_root_dir=config.output.root,
    )
    try:
        trainer.fit(module, loader, ckpt_path=config.get("resume_checkpoint"))
    finally:
        if clearml_task is not None:
            clearml_task.close()


if __name__ == "__main__":
    main()
