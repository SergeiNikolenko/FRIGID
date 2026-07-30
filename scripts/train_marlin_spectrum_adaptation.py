#!/usr/bin/env python3
# ruff: noqa: E402
"""Adapt a MARLIN decoder to structure-disjoint predicted spectrum fingerprints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import lightning as L
import torch
from clearml import Task
from omegaconf import OmegaConf

from marlin.clearml_metrics import ClearMLTrainingMetrics
from marlin.model import MarlinDecoderConfig
from marlin.periodic_evaluation import PeriodicMolecularEvaluation
from marlin.tokenizer import load_safe_tokenizer, validate_safe_tokenizer
from marlin.training import (
    MarlinCollator,
    MarlinLightningModule,
    MarlinSpectrumFingerprintDataset,
)
from marlin.warm_start import load_marlin_decoder_weights, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--fingerprints", type=Path, required=True)
    parser.add_argument("--fingerprint-key", default="probs")
    parser.add_argument("--fingerprint-threshold", type=float, default=0.90)
    parser.add_argument("--exclude-inchikeys", type=Path, required=True)
    parser.add_argument(
        "--exclude-metadata",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--validation-metadata", type=Path, required=True)
    parser.add_argument("--validation-fingerprints", type=Path, required=True)
    parser.add_argument("--validation-fingerprint-key", default="probs")
    parser.add_argument(
        "--validation-fingerprint-threshold",
        type=float,
        default=0.95,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--evaluation-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--evaluation-spectra", type=int, default=4)
    parser.add_argument("--evaluation-candidates", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulate-grad-batches", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--cross-attention-only-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str, list[str]]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
    ).splitlines()
    return commit, dirty


def _decoder_config(checkpoint: dict) -> MarlinDecoderConfig:
    config = MarlinDecoderConfig(**checkpoint["hyper_parameters"]["config"])
    state = checkpoint["state_dict"]
    if "decoder.conditioner.fingerprint.layer_norm.weight" not in state:
        config = replace(config, fingerprint_layer_norm=False)
    return config


def _clearml_task(args: argparse.Namespace, config: dict) -> Task:
    task = Task.init(
        project_name="MARLIN clean-room reproduction",
        task_name="marlin-spectrum-fingerprint-adaptation",
        tags=[
            "MARLIN",
            "autoresearch",
            "spectrum-fingerprint-adaptation",
            "DreaMS",
            "non-oracle",
            "structure-disjoint",
            "molecular-gates",
        ],
        reuse_last_task_id=False,
        output_uri=False,
        auto_connect_streams=False,
        auto_connect_frameworks={"pytorch": True, "tensorboard": True},
    )
    task.connect(
        config,
        name="resolved_config",
        ignore_remote_overrides=True,
    )
    return task


def main() -> None:
    args = parse_args()
    if min(
        args.max_steps,
        args.evaluation_interval,
        args.checkpoint_interval,
        args.evaluation_spectra,
        args.evaluation_candidates,
        args.batch_size,
        args.accumulate_grad_batches,
    ) <= 0:
        raise ValueError("training and evaluation sizes must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    observed_checkpoint_sha256 = _sha256(args.checkpoint)
    if observed_checkpoint_sha256 != args.checkpoint_sha256:
        raise ValueError(
            "checkpoint SHA-256 mismatch: "
            f"{observed_checkpoint_sha256} != {args.checkpoint_sha256}"
        )
    commit, dirty = _git_state()
    if dirty:
        raise RuntimeError(f"refusing adaptation from dirty Git state: {dirty}")

    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    tokenizer = load_safe_tokenizer(args.tokenizer)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    decoder_config = _decoder_config(checkpoint)
    del checkpoint
    validate_safe_tokenizer(
        tokenizer,
        expected_vocab_size=decoder_config.vocab_size,
        expected_special_token_ids={
            "unk": 0,
            "bos": 1,
            "eos": decoder_config.eos_token_id,
            "pad": decoder_config.pad_token_id,
            "mask": decoder_config.mask_token_id,
        },
    )

    dataset = MarlinSpectrumFingerprintDataset(
        args.metadata,
        args.fingerprints,
        tokenizer,
        fingerprint_key=args.fingerprint_key,
        threshold=args.fingerprint_threshold,
        max_length=decoder_config.max_length,
        exclude_inchikeys=args.exclude_inchikeys,
        exclude_metadata_csvs=tuple(args.exclude_metadata),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=MarlinCollator(
            tokenizer,
            max_length=decoder_config.max_length,
            fingerprint_bits=decoder_config.fingerprint_bits,
            exclude_inchikeys=args.exclude_inchikeys,
        ),
    )

    module = MarlinLightningModule(
        decoder_config,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        noise_probability=0.0,
        noise_min_fraction=0.1,
        noise_max_fraction=0.3,
        ema_decay=0.9999,
        metric_interval=25,
        conditioning_only_steps=0,
        cross_attention_only_steps=args.cross_attention_only_steps,
        adapt_fingerprint=True,
    )
    start_report = load_marlin_decoder_weights(
        module.decoder,
        args.checkpoint,
        architecture_upgrade=False,
        use_ema=False,
    )
    module.reset_ema()

    args.output_dir.mkdir(parents=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    resolved = {
        "seed": args.seed,
        "source_checkpoint": str(args.checkpoint),
        "source_checkpoint_sha256": args.checkpoint_sha256,
        "training_metadata": str(args.metadata),
        "training_fingerprints": str(args.fingerprints),
        "training_fingerprint_key": args.fingerprint_key,
        "training_fingerprint_threshold": args.fingerprint_threshold,
        "excluded_metadata": [str(path) for path in args.exclude_metadata],
        "training_rows_after_exclusions": len(dataset),
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "cross_attention_only_steps": args.cross_attention_only_steps,
        "global_batch_size": args.batch_size * args.accumulate_grad_batches,
        "evaluation": {
            "metadata": str(args.validation_metadata),
            "fingerprints": str(args.validation_fingerprints),
            "fingerprint_key": args.validation_fingerprint_key,
            "threshold": args.validation_fingerprint_threshold,
            "max_spectra": args.evaluation_spectra,
            "candidates": args.evaluation_candidates,
        },
    }
    manifest = {
        "schema_version": 1,
        "kind": "MARLIN spectrum-fingerprint adaptation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "clean_room_reproduction": True,
        "selection_split_only": True,
        "locked_test_used": False,
        "config": resolved,
        "inputs": {
            str(path): {"sha256": sha256_file(path)}
            for path in (
                args.checkpoint,
                args.tokenizer,
                args.metadata,
                args.fingerprints,
                args.exclude_inchikeys,
                *args.exclude_metadata,
                args.validation_metadata,
                args.validation_fingerprints,
            )
        },
        "initialization": start_report,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    config = OmegaConf.create(
        {
            "evaluation": {
                "enabled": True,
                "interval_steps": args.evaluation_interval,
                "metadata": str(args.validation_metadata),
                "fingerprints": str(args.validation_fingerprints),
                "fingerprint_key": args.validation_fingerprint_key,
                "threshold": args.validation_fingerprint_threshold,
                "use_ema": False,
                "lane": "dreams",
                "max_spectra": args.evaluation_spectra,
                "candidates": args.evaluation_candidates,
                "diversity_dropout": 0.3,
                "temperature": 1.0,
                "ppm_tolerance": 10.0,
                "seed": args.seed,
            },
            "data": {"tokenizer_file": str(args.tokenizer)},
            "output": {
                "root": str(args.output_dir),
                "checkpoints": str(checkpoint_dir),
                "checkpoint_interval": args.checkpoint_interval,
            },
        }
    )
    clearml_task = _clearml_task(args, resolved)
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{step}",
        every_n_train_steps=args.checkpoint_interval,
        save_top_k=1,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        log_every_n_steps=25,
        callbacks=[
            checkpoint_callback,
            PeriodicMolecularEvaluation(
                config,
                project_root=PROJECT_ROOT,
                clearml_task=clearml_task,
            ),
            ClearMLTrainingMetrics(clearml_task),
        ],
        default_root_dir=args.output_dir,
    )
    try:
        trainer.fit(module, loader)
    finally:
        clearml_task.close()


if __name__ == "__main__":
    main()
