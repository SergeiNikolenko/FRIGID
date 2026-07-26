#!/usr/bin/env python3
# ruff: noqa: E402
"""Train the clean-room MARLIN block-diffusion decoder."""

from __future__ import annotations

import hashlib
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
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import datasets
import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from marlin.dataset import verify_snapshot_manifest
from marlin.distillation import (
    FRIGID_DISTILLED_MARLIN_MODE,
    FrigidDistillationSettings,
    FrozenFrigidTeacherCheckpoint,
)
from marlin.model import MarlinDecoderConfig
from marlin.tokenizer import load_safe_tokenizer, validate_safe_tokenizer
from marlin.training import (
    ClearMLScalarCallback,
    MarlinCollator,
    MarlinLightningModule,
    MarlinMolecularValidationCallback,
    MarlinMetadataDataset,
    MarlinTrainingFilter,
)
from marlin.warm_start import load_frigid_decoder


STRICT_MARLIN_MODE = "strict_marlin"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def adaptation_mode(config: DictConfig) -> str:
    """Return the explicit training lane and reject ambiguous configurations."""
    adaptation = config.get("adaptation")
    if adaptation is None or not adaptation.get("mode"):
        raise ValueError("adaptation.mode must be explicit")
    mode = str(adaptation.mode)
    if mode not in {STRICT_MARLIN_MODE, FRIGID_DISTILLED_MARLIN_MODE}:
        raise ValueError(f"unsupported adaptation.mode: {mode!r}")
    if mode == STRICT_MARLIN_MODE:
        incompatible = {
            "adaptation.teacher_checkpoint": adaptation.get("teacher_checkpoint"),
            "adaptation.teacher_sha256": adaptation.get("teacher_sha256"),
        }
        present = [name for name, value in incompatible.items() if value]
        if present:
            raise ValueError(
                "strict_marlin forbids FRIGID teacher fields: "
                + ", ".join(present)
            )
    return mode


def build_distillation(
    config: DictConfig,
    tokenizer,
    special_token_ids: dict[str, int],
) -> tuple[
    FrigidDistillationSettings | None,
    FrozenFrigidTeacherCheckpoint | None,
]:
    """Build the lazy per-rank teacher specification for the opt-in lane."""
    mode = adaptation_mode(config)
    if mode == STRICT_MARLIN_MODE:
        return None, None

    required = {
        "frigid_warm_start_checkpoint": config.get(
            "frigid_warm_start_checkpoint"
        ),
        "frigid_warm_start_sha256": config.get("frigid_warm_start_sha256"),
        "adaptation.teacher_checkpoint": config.adaptation.get(
            "teacher_checkpoint"
        ),
        "adaptation.teacher_sha256": config.adaptation.get("teacher_sha256"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(
            "FRIGID-distilled MARLIN requires " + ", ".join(missing)
        )

    settings = FrigidDistillationSettings(
        mode=mode,
        block_width_override=int(config.adaptation.block_width_override),
        attention_mode=str(config.adaptation.get("attention_mode", "frigid_full")),
        temperature=float(config.adaptation.temperature),
        kl_weight=float(config.adaptation.kl_weight),
        use_isotope=bool(config.adaptation.get("use_isotope", False)),
    )
    teacher_checkpoint = FrozenFrigidTeacherCheckpoint(
        checkpoint_path=str(config.adaptation.teacher_checkpoint),
        expected_sha256=str(config.adaptation.teacher_sha256),
        expected_tokenizer_vocab=dict(tokenizer.get_vocab()),
        expected_special_token_ids={
            name: int(token_id) for name, token_id in special_token_ids.items()
        },
    )
    return settings, teacher_checkpoint


def initialization_source(config: DictConfig) -> str | None:
    """Resolve one effective initialization source.

    The distilled lane keeps the hashed FRIGID warm-start in its configuration
    as immutable provenance. On a full Lightning resume, however, weights and
    optimizer state must come only from the resume checkpoint.
    """
    mode = adaptation_mode(config)
    sources = {
        "resume_checkpoint": config.get("resume_checkpoint"),
        "resume_weights_only_checkpoint": config.get(
            "resume_weights_only_checkpoint"
        ),
        "frigid_warm_start_checkpoint": config.get(
            "frigid_warm_start_checkpoint"
        ),
    }
    if mode == FRIGID_DISTILLED_MARLIN_MODE and (
        sources["resume_checkpoint"]
        or sources["resume_weights_only_checkpoint"]
    ):
        sources["frigid_warm_start_checkpoint"] = None
    selected = [name for name, value in sources.items() if value]
    if len(selected) > 1:
        raise ValueError(
            "choose exactly one initialization source; got "
            + ", ".join(selected)
        )
    return selected[0] if selected else None


def write_run_manifest(config: DictConfig, tokenizer_sha256: str) -> dict:
    """Persist the immutable training inputs and execution environment."""
    commit, dirty = git_state()
    if dirty:
        raise RuntimeError(f"refusing canonical training from dirty git state: {dirty}")
    mode = adaptation_mode(config)
    strict_reproduction = mode == STRICT_MARLIN_MODE
    manifest = {
        "schema_version": 1,
        "kind": (
            "MARLIN clean-room decoder training"
            if strict_reproduction
            else "FRIGID-distilled MARLIN stage-0 adaptation"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "adaptation_mode": mode,
        "strict_reproduction": strict_reproduction,
        "clean_room_reproduction": strict_reproduction,
        "author_code_available_at_start": False,
        "git_commit": commit,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "config": OmegaConf.to_container(config, resolve=True),
        "inputs": {
            "tokenizer_sha256": tokenizer_sha256,
            "nplib1_test_exclusions_sha256": sha256_file(
                config.data.exclude_inchikeys
            ),
            "training_length_audit_sha256": sha256_file(
                config.data.length_audit_manifest
            ),
            "dataset": str(config.data.dataset),
            "dataset_revision": str(config.data.revision),
            "training_snapshot_manifest": str(config.data.snapshot_manifest),
            "training_snapshot_manifest_sha256": sha256_file(
                config.data.snapshot_manifest
            ),
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
        auto_connect_streams=False,
        auto_connect_frameworks={"pytorch": True, "tensorboard": True},
        auto_resource_monitoring={
            "report_frequency_sec": 5.0,
            "first_report_sec": 5.0,
            "wait_for_first_iteration_to_start_sec": 5.0,
            "max_wait_for_first_iteration_to_start_sec": 5.0,
        },
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


def load_decoder_weights_only(module: MarlinLightningModule, checkpoint_path: str | Path) -> None:
    """Load decoder weights from a Lightning checkpoint without optimizer state."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    decoder_state = {
        key.removeprefix("decoder."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("decoder.")
    }
    module.decoder.load_state_dict(decoder_state, strict=True)
    module.reset_ema()


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
    distillation, frigid_teacher = build_distillation(
        config,
        tokenizer,
        special_token_ids,
    )
    metadata_csv = config.data.get("metadata_csv")
    local_shards = None
    if metadata_csv is None:
        length_audit = json.loads(Path(config.data.length_audit_manifest).read_text())
        if sha256_file(config.data.length_audit_manifest) != config.data.length_audit_sha256:
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
        if snapshot_manifest_sha256 != sha256_file(config.data.snapshot_manifest):
            raise ValueError("training snapshot manifest changed during verification")
    write_run_manifest(config, tokenizer_sha256)
    module = MarlinLightningModule(
        decoder_config,
        learning_rate=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
        noise_probability=config.training.noise_probability,
        noise_min_fraction=config.training.noise_min_fraction,
        noise_max_fraction=config.training.noise_max_fraction,
        ema_decay=config.training.ema_decay,
        metric_interval=config.training.metric_interval,
        eos_loss_weight=config.training.get("eos_loss_weight", 1.0),
        eos_mask_probability=config.training.get("eos_mask_probability", 0.0),
        balanced_token_loss_alpha=config.training.get("balanced_token_loss_alpha", 0.0),
        token_loss_weight_max=config.training.get("token_loss_weight_max", 20.0),
        full_sequence_mask_probability=config.training.get(
            "full_sequence_mask_probability", 0.0
        ),
        distillation=distillation,
        frigid_teacher=frigid_teacher,
    )
    selected_source = initialization_source(config)
    if selected_source == "frigid_warm_start_checkpoint":
        if not config.get("frigid_warm_start_sha256"):
            raise ValueError(
                "frigid_warm_start_sha256 is required with "
                "frigid_warm_start_checkpoint"
            )
        report = load_frigid_decoder(
            module.decoder,
            config.frigid_warm_start_checkpoint,
            expected_sha256=config.get("frigid_warm_start_sha256"),
        )
        module.reset_ema()
        report["mode"] = "frigid_architecture_compatible"
        report_path = Path(config.output.root) / "warm_start.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    elif selected_source == "resume_weights_only_checkpoint":
        load_decoder_weights_only(module, config.resume_weights_only_checkpoint)
        report = {
            "source": str(config.resume_weights_only_checkpoint),
            "mode": "lightning_decoder_weights_only",
        }
        report_path = Path(config.output.root) / "warm_start.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    if metadata_csv is None:
        dataset = datasets.load_dataset(
            "parquet",
            data_files={"train": local_shards},
            split="train",
            streaming=True,
            cache_dir=config.data.hf_cache_dir,
        ).filter(
            MarlinTrainingFilter(
                tokenizer,
                decoder_config.max_length,
                config.data.exclude_inchikeys,
            ),
        )
        dataset = dataset.shuffle(seed=config.seed, buffer_size=config.data.shuffle_buffer)
    else:
        dataset = MarlinMetadataDataset(
            metadata_csv,
            tokenizer,
            max_length=decoder_config.max_length,
            exclude_inchikeys=config.data.get("metadata_exclude_inchikeys"),
        )
    collator = MarlinCollator(
        tokenizer,
        max_length=decoder_config.max_length,
        fingerprint_bits=decoder_config.fingerprint_bits,
        exclude_inchikeys=config.data.exclude_inchikeys,
        include_formula=distillation is not None,
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
    callbacks: list[L.Callback] = [checkpoint]
    if clearml_task is not None:
        callbacks.append(
            ClearMLScalarCallback(
                clearml_task,
                every_n_steps=config.training.metric_interval,
            )
        )
    molecular_validation_csv = config.training.get("molecular_validation_csv")
    if molecular_validation_csv:
        callbacks.append(
            MarlinMolecularValidationCallback(
                tokenizer,
                molecular_validation_csv,
                output_dir=config.output.root,
                fingerprint_bits=decoder_config.fingerprint_bits,
                every_n_steps=config.training.molecular_validation_interval,
                samples=config.training.molecular_validation_samples,
                candidates=config.training.molecular_validation_candidates,
                temperature=config.training.molecular_validation_temperature,
                clearml_task=clearml_task,
            )
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
        callbacks=callbacks,
        default_root_dir=config.output.root,
    )
    try:
        trainer.fit(module, loader, ckpt_path=config.get("resume_checkpoint"))
    finally:
        if clearml_task is not None:
            clearml_task.close()


if __name__ == "__main__":
    main()
