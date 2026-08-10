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

from marlin.checkpoint_selection import CheckpointSelector
from marlin.clearml_metrics import ClearMLTrainingMetrics
from marlin.model import MarlinDecoderConfig
from marlin.periodic_evaluation import PeriodicMolecularEvaluation
from marlin.tokenizer import load_safe_tokenizer, validate_safe_tokenizer
from marlin.training import (
    MarlinCollator,
    MarlinLightningModule,
    MarlinSpectrumFingerprintDataset,
    PeriodicHeldOutLoss,
    holdout_split_report,
    structure_disjoint_holdout,
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
    parser.add_argument(
        "--soft-fingerprint",
        action="store_true",
        help="train on DreaMS probability amplitudes above the threshold",
    )
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
        "--evaluation-manifest",
        type=Path,
        default=PROJECT_ROOT
        / "configs/benchmarks/nplib1_v1/nplib1_val_full396_v1.tsv",
        help=(
            "held-out molecular panel. The 32-spectrum micro panel is retired: "
            "its floor is one molecule in 32 and it swings by up to 0.12 between "
            "neighbouring checkpoints, which is where several withdrawn claims "
            "in docs/MARLIN_STATUS_REPORT.md came from"
        ),
    )
    parser.add_argument(
        "--validation-fingerprint-threshold",
        type=float,
        default=0.95,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--evaluation-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--evaluation-spectra", type=int, default=396)
    parser.add_argument(
        "--evaluation-candidates",
        type=int,
        default=8,
        help=(
            "screening budget per spectrum; 8 is what the sharded cost figures "
            "in docs/MARLIN_STATUS_REPORT.md were measured at"
        ),
    )
    parser.add_argument(
        "--evaluation-shards",
        type=int,
        default=16,
        help=(
            "split the panel across this many single-core evaluator processes. "
            "The constrained decoder is CPU bound, so 16 shards take the "
            "803-spectrum split from about 17 h to about 1.1 h; 1 restores the "
            "single-process path"
        ),
    )
    parser.add_argument(
        "--validation-loss-fraction",
        type=float,
        default=0.0,
        help=(
            "hold out this fraction of the adaptation structures from the "
            "optimizer and log the masked-diffusion objective on them at every "
            "evaluation interval; 0 disables the held-out loss"
        ),
    )
    parser.add_argument(
        "--validation-loss-split-seed",
        type=int,
        default=0,
        help=(
            "seed of the structure hash that carves the held-out slice. Kept "
            "separate from --seed so a multi-seed protocol scores every seed on "
            "the same held-out molecules"
        ),
    )
    parser.add_argument(
        "--select-best-checkpoint",
        action="store_true",
        help=(
            "record the best checkpoint on the held-out molecular panel and "
            "enable early stopping; without it a run keeps its historical "
            "behaviour of no selection signal at all"
        ),
    )
    parser.add_argument(
        "--selection-metric",
        default="candidate_return_rate",
        help=(
            "held-out metric that drives selection. Exact@k is refused: it was "
            "flat at 0.0312 from step 20,000 to 70,000 while candidate return, "
            "uniqueness and mass validity all halved"
        ),
    )
    parser.add_argument(
        "--selection-patience",
        type=int,
        default=0,
        help=(
            "stop after this many periodic evaluations without improvement; "
            "0 records the best checkpoint but never stops"
        ),
    )
    parser.add_argument("--selection-min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulate-grad-batches", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--cross-attention-only-steps", type=int, default=100)
    parser.add_argument("--noise-probability", type=float, default=0.5)
    parser.add_argument("--noise-min-fraction", type=float, default=0.1)
    parser.add_argument("--noise-max-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mass-reachability-prune",
        action="store_true",
        help=(
            "fold mass reachability into the periodic evaluation's syntax mask; "
            "measured at 5x candidate return and 3.8x mass validity on the "
            "32-spectrum panel"
        ),
    )
    parser.add_argument(
        "--forbid-isotope-tokens",
        action="store_true",
        help=(
            "withhold periodic-evaluation support from the 479 isotope-labelled "
            "vocabulary tokens, which no adaptation target uses and the "
            "monoisotopic mass shell can never accept"
        ),
    )
    parser.add_argument(
        "--restrict-organic-elements",
        action="store_true",
        help=(
            "withhold periodic-evaluation support from tokens introducing an "
            "element outside CHNOPS and the halogens, which no adaptation "
            "target uses"
        ),
    )
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


def resolve_clearml_output_uri() -> str | bool:
    """Where ClearML should store checkpoints.

    On SLURM the run directory is already on shared storage, so uploading is
    pure overhead and the default stays disabled. A queued worker has its own
    filesystem, so the checkpoints it writes are unreachable from anywhere
    else; set MARLIN_CLEARML_OUTPUT_URI to the files server in that case.
    """
    return os.environ.get("MARLIN_CLEARML_OUTPUT_URI", "").strip() or False


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
        output_uri=resolve_clearml_output_uri(),
        auto_connect_streams=False,
        auto_connect_frameworks={"pytorch": True, "tensorboard": True},
    )
    task.connect(
        config,
        name="resolved_config",
        ignore_remote_overrides=True,
    )
    return task


def validate_args(args: argparse.Namespace) -> None:
    """Reject configurations before any weights or datasets are touched."""
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
    if not 0.0 <= args.noise_probability <= 1.0:
        raise ValueError("noise probability must be in [0, 1]")
    if not 0.0 <= args.noise_min_fraction <= args.noise_max_fraction <= 1.0:
        raise ValueError("noise fractions must satisfy 0 <= min <= max <= 1")
    if args.evaluation_shards < 1:
        raise ValueError("--evaluation-shards must be at least 1")
    if args.evaluation_spectra > 128 and args.evaluation_shards < 2:
        raise ValueError(
            "a full validation panel needs sharding: at one process the "
            f"{args.evaluation_spectra}-spectrum panel does not fit an "
            "evaluation interval; pass --evaluation-shards"
        )
    if not 0.0 <= args.validation_loss_fraction < 0.5:
        raise ValueError("--validation-loss-fraction must be in [0, 0.5)")
    if args.selection_patience < 0:
        raise ValueError("--selection-patience must be non-negative")
    if args.select_best_checkpoint:
        # Reject an unusable selection metric here rather than inside the
        # callback, which is built after the run directory and the ClearML task.
        CheckpointSelector(
            metric=args.selection_metric,
            patience=args.selection_patience,
            min_delta=args.selection_min_delta,
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
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
        preserve_probabilities=args.soft_fingerprint,
    )
    collator = MarlinCollator(
        tokenizer,
        max_length=decoder_config.max_length,
        fingerprint_bits=decoder_config.fingerprint_bits,
        exclude_inchikeys=args.exclude_inchikeys,
        allow_soft_fingerprints=args.soft_fingerprint,
    )
    # The held-out slice is carved out of the adaptation set by connectivity
    # block, not out of the validation split: the validation split is exactly the
    # 396-spectrum molecular panel, so a loss measured on it would consume the
    # panel it is supposed to complement.
    train_indices, holdout_indices = structure_disjoint_holdout(
        dataset.rows,
        fraction=args.validation_loss_fraction,
        seed=args.validation_loss_split_seed,
    )
    holdout_report = holdout_split_report(
        dataset.rows, train_indices, holdout_indices
    )
    train_dataset = (
        dataset
        if not holdout_indices
        else torch.utils.data.Subset(dataset, train_indices)
    )
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    holdout_loader = (
        torch.utils.data.DataLoader(
            torch.utils.data.Subset(dataset, holdout_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
        )
        if holdout_indices
        else None
    )

    module = MarlinLightningModule(
        decoder_config,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        noise_probability=args.noise_probability,
        noise_min_fraction=args.noise_min_fraction,
        noise_max_fraction=args.noise_max_fraction,
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
        "training_soft_fingerprint": args.soft_fingerprint,
        "excluded_metadata": [str(path) for path in args.exclude_metadata],
        "training_rows_after_exclusions": len(dataset),
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "cross_attention_only_steps": args.cross_attention_only_steps,
        "symmetric_fingerprint_noise": {
            "probability": args.noise_probability,
            "min_fraction": args.noise_min_fraction,
            "max_fraction": args.noise_max_fraction,
            "equal_drop_add": True,
        },
        "global_batch_size": args.batch_size * args.accumulate_grad_batches,
        "held_out_loss": {
            "enabled": bool(holdout_indices),
            "fraction": args.validation_loss_fraction,
            "split_seed": args.validation_loss_split_seed,
            "split_key": "inchikey_first_block",
            "source": "adaptation set, not the molecular validation panel",
            "fingerprint_corruption": False,
            **holdout_report,
        },
        "evaluation": {
            "metadata": str(args.validation_metadata),
            "fingerprints": str(args.validation_fingerprints),
            "fingerprint_key": args.validation_fingerprint_key,
            "threshold": args.validation_fingerprint_threshold,
            "spec_manifest": str(args.evaluation_manifest),
            "max_spectra": args.evaluation_spectra,
            "candidates": args.evaluation_candidates,
            "shards": args.evaluation_shards,
            "sample_tokens": True,
            "soft_fingerprint": args.soft_fingerprint,
            "selection": {
                "enabled": bool(args.select_best_checkpoint),
                "metric": args.selection_metric,
                "patience": args.selection_patience,
                "min_delta": args.selection_min_delta,
            },
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
                args.evaluation_manifest,
            )
        },
        "initialization": start_report,
    }
    manifest_path = args.output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    config = OmegaConf.create(
        {
            "evaluation": {
                "enabled": True,
                "interval_steps": args.evaluation_interval,
                "metadata": str(args.validation_metadata),
                "fingerprints": str(args.validation_fingerprints),
                "fingerprint_key": args.validation_fingerprint_key,
                "threshold": args.validation_fingerprint_threshold,
                "spec_manifest": str(args.evaluation_manifest),
                "use_ema": False,
                "lane": "dreams",
                "max_spectra": args.evaluation_spectra,
                "candidates": args.evaluation_candidates,
                "shards": args.evaluation_shards,
                "selection": {
                    "enabled": bool(args.select_best_checkpoint),
                    "metric": args.selection_metric,
                    "patience": args.selection_patience,
                    "min_delta": args.selection_min_delta,
                },
                "diversity_dropout": 0.3,
                "temperature": 1.0,
                "sample_tokens": True,
                "soft_fingerprint": args.soft_fingerprint,
                "mass_reachability_prune": args.mass_reachability_prune,
                "forbid_isotope_tokens": args.forbid_isotope_tokens,
                "restrict_organic_elements": args.restrict_organic_elements,
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
    manifest["clearml"] = {
        "task_id": clearml_task.id,
        "offline_mode": os.environ.get("CLEARML_OFFLINE_MODE", "0"),
        "cache_dir": os.environ.get("CLEARML_CACHE_DIR"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{step}",
        every_n_train_steps=args.checkpoint_interval,
        # Without a monitor, save_top_k=1 makes Lightning delete the previous
        # periodic checkpoint, so an adaptation run keeps no history to compare
        # or fall back to.
        save_top_k=-1,
    )
    callbacks = [checkpoint_callback]
    if holdout_loader is not None:
        # Before the molecular panel, so the cheap held-out signal is logged even
        # when the panel is still running or fails.
        callbacks.append(
            PeriodicHeldOutLoss(
                holdout_loader,
                interval_steps=args.evaluation_interval,
                seed=args.validation_loss_split_seed,
                clearml_task=clearml_task,
            )
        )
    callbacks.extend(
        [
            PeriodicMolecularEvaluation(
                config,
                project_root=PROJECT_ROOT,
                clearml_task=clearml_task,
            ),
            ClearMLTrainingMetrics(clearml_task),
        ]
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=1.0,
        log_every_n_steps=25,
        callbacks=callbacks,
        default_root_dir=args.output_dir,
    )
    try:
        trainer.fit(module, loader)
    finally:
        clearml_task.close()


if __name__ == "__main__":
    main()
