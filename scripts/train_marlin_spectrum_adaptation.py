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

from marlin.checkpoint_selection import (
    PANEL_SELECTION_METRICS,
    PROBE_SELECTION_METRICS,
    CheckpointSelector,
)
from marlin.clearml_metrics import ClearMLTrainingMetrics
from marlin.conditioning_probe import (
    PeriodicConditioningProbe,
    probe_batches_from_paths,
)
from marlin.gradient_diagnostics import EmaDivergence, GradientDiagnostics
from marlin.lr_schedule import (
    ADAMW_SECOND_MOMENT_WINDOW,
    RELEASED_TERMINAL_LEARNING_RATE,
    derive_peak_learning_rate,
    schedule_displacement,
)
from marlin.model import (
    LOSS_REDUCTIONS,
    TIME_SAMPLING_MODES,
    MarlinDecoderConfig,
)
from marlin.periodic_evaluation import PeriodicMolecularEvaluation
from marlin.tokenizer import load_safe_tokenizer, validate_safe_tokenizer
from marlin.training import (
    LR_SCHEDULES,
    MarlinCollator,
    MarlinLightningModule,
    MarlinSpectrumFingerprintDataset,
    PeriodicHeldOutLoss,
    holdout_split_report,
    structure_disjoint_holdout,
)
from marlin.warm_start import load_marlin_decoder_weights, sha256_file


# The threshold the reported clipping rate is measured against; keep the
# trainer and the diagnostics reading the same number.
GRADIENT_CLIP_VAL = 1.0


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
        "--gradient-diagnostics-interval",
        type=int,
        default=0,
        help=(
            "log pre-clip gradient norms per pathway (fingerprint conditioner, "
            "cross-attention stack, backbone), the global norm, the clipping "
            "rate and the EMA-to-live distance every this many steps; 0 keeps "
            "the historical single global norm every 50 steps and nothing else"
        ),
    )
    parser.add_argument(
        "--conditioning-probe-interval",
        type=int,
        default=0,
        help=(
            "run the fixed conditioning probe every this many steps: "
            "teacher-forced top-1 under the true versus the predicted "
            "fingerprint, and the fraction of the probe whose loss is "
            "explained by the conditioning-free prior; 0 disables it"
        ),
    )
    parser.add_argument("--conditioning-probe-metadata", type=Path, default=None)
    parser.add_argument("--conditioning-probe-fingerprints", type=Path, default=None)
    parser.add_argument("--conditioning-probe-fingerprint-key", default="probs")
    parser.add_argument(
        "--conditioning-probe-threshold",
        type=float,
        default=None,
        help="defaults to --validation-fingerprint-threshold, the gate inference uses",
    )
    parser.add_argument("--conditioning-probe-size", type=int, default=64)
    parser.add_argument("--conditioning-probe-batch-size", type=int, default=8)
    parser.add_argument(
        "--conditioning-probe-seed",
        type=int,
        default=0,
        help=(
            "seed of the probe's structure hash and mask draw. Kept separate "
            "from --seed so every arm of a multi-seed protocol is probed on the "
            "same molecules at the same positions"
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
    parser.add_argument(
        "--probe-early-stopping-metric",
        default=None,
        choices=(None, *PROBE_SELECTION_METRICS),
        help=(
            "stop the run when this conditioning-probe metric stops improving. "
            "'probe_top1_predicted' is teacher-forced per-token top-1 under the "
            "fingerprint inference supplies, on structures the optimizer never "
            "sees. Needs --conditioning-probe-interval. Off by default, which "
            "is the historical behaviour: no run in this lineage has ever had a "
            "stopping criterion, and the corpus is replayed 3,846 times over "
            "100,000 steps"
        ),
    )
    parser.add_argument(
        "--probe-early-stopping-patience",
        type=int,
        default=3,
        help=(
            "probes without improvement before stopping. MS-BART uses 3 at an "
            "every-200-step cadence (docs/TRAINING_RECIPE_FINDINGS.md:48)"
        ),
    )
    parser.add_argument(
        "--probe-early-stopping-min-delta",
        type=float,
        default=0.0,
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulate-grad-batches", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help=(
            "the optimizer's base rate, and the PEAK when --lr-schedule is "
            "warmup_cosine"
        ),
    )
    parser.add_argument(
        "--lr-schedule",
        choices=LR_SCHEDULES,
        default="constant",
        help=(
            "'constant' is the historical recipe: no schedule at all, resuming "
            "weights annealed to 5.2697058404552555e-08 at a flat 1e-5, which "
            "is 190x the rate they were left at. 'warmup_cosine' warms up over "
            "--lr-warmup-steps and anneals to --lr-min; see "
            "src/marlin/lr_schedule.py for the peak derivation"
        ),
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=ADAMW_SECOND_MOMENT_WINDOW,
        help=(
            "default 1000 = AdamW's second-moment averaging window at "
            "beta2=0.999; the optimizer state is not restored, so the peak is "
            "not meaningful before that window has filled"
        ),
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=RELEASED_TERMINAL_LEARNING_RATE,
        help=(
            "where the cosine ends; defaults to the rate the released "
            "checkpoint's own schedule left its weights at"
        ),
    )
    parser.add_argument(
        "--derive-learning-rate",
        action="store_true",
        help=(
            "ignore --learning-rate and set the peak from the displacement "
            "budget of the released run's final decade of learning rate, given "
            "--max-steps, --lr-warmup-steps and --lr-min. Requires "
            "--lr-schedule warmup_cosine"
        ),
    )
    parser.add_argument(
        "--time-sampling",
        choices=TIME_SAMPLING_MODES,
        default="per_block_iid",
        help=(
            "'per_block_iid' is the historical recipe: one diffusion time per "
            "(example, block), so a 256-token row draws 32 of them. "
            "'per_sequence_antithetic' is what the released checkpoint was "
            "trained with: one stratified t per sequence over [eps, 1] "
            "(src/dlm/model.py:222-223)"
        ),
    )
    parser.add_argument("--time-sampling-eps", type=float, default=1e-3)
    parser.add_argument(
        "--loss-reduction",
        choices=LOSS_REDUCTIONS,
        default="block_mean",
        help=(
            "'block_mean' is the historical recipe: per-example sum over "
            "ceil(n_valid/8), about 8x a token mean. 'token_mean' is the "
            "checkpoint's global_mean_loss, sum over valid tokens divided by "
            "their count (src/dlm/model.py:953)"
        ),
    )
    parser.add_argument(
        "--fp32-forward",
        action="store_true",
        help=(
            "compute the training forward in fp32 under a bf16-mixed trainer, "
            "which is what the released model does (src/dlm/model.py:949)"
        ),
    )
    parser.add_argument("--cross-attention-only-steps", type=int, default=100)
    parser.add_argument(
        "--context-corruption-probability",
        type=float,
        default=0.0,
        help=(
            "probability that a training example has part of its clean-stream "
            "context replaced by the model's own runner-up tokens, with the "
            "cross-entropy targets left gold. At sampling time the clean stream "
            "is the committed prefix, so a wrong committed token is context the "
            "training objective never produces; 0 keeps the historical objective"
        ),
    )
    parser.add_argument(
        "--context-corruption-warmup-steps",
        type=int,
        default=1000,
        help="linear ramp of the corruption probability from 0",
    )
    parser.add_argument("--context-corruption-min-fraction", type=float, default=0.05)
    parser.add_argument("--context-corruption-max-fraction", type=float, default=0.25)
    parser.add_argument(
        "--restoration-loss-weight",
        type=float,
        default=0.0,
        help=(
            "coefficient of the extra cross-entropy on corrupted, unmasked "
            "positions. It is not part of the absorbing NELBO, so it is kept "
            "as a separate mean with its own coefficient"
        ),
    )
    parser.add_argument("--noise-probability", type=float, default=0.5)
    parser.add_argument(
        "--fingerprint-noise-mode",
        choices=("symmetric", "dropout"),
        default="symmetric",
        help=(
            "how the conditioning fingerprint is corrupted during training. "
            "'symmetric' moves an on-bit to an off position, keeping the "
            "cardinality but inventing bits; 'dropout' only removes on-bits, "
            "which is what inference diversity does to the vector the decoder "
            "is given"
        ),
    )
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
    if not 0.0 <= args.context_corruption_probability <= 1.0:
        raise ValueError("context corruption probability must be in [0, 1]")
    if not (
        0.0
        <= args.context_corruption_min_fraction
        <= args.context_corruption_max_fraction
        <= 1.0
    ):
        raise ValueError(
            "context corruption fractions must satisfy 0 <= min <= max <= 1"
        )
    if args.context_corruption_warmup_steps < 0:
        raise ValueError("--context-corruption-warmup-steps must be non-negative")
    if args.restoration_loss_weight < 0:
        raise ValueError("--restoration-loss-weight must be non-negative")
    if args.restoration_loss_weight and not args.context_corruption_probability:
        raise ValueError(
            "a restoration loss without context corruption has no corrupted "
            "positions to score"
        )
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
    if args.gradient_diagnostics_interval < 0:
        raise ValueError("--gradient-diagnostics-interval must be non-negative")
    if args.conditioning_probe_interval < 0:
        raise ValueError("--conditioning-probe-interval must be non-negative")
    if args.conditioning_probe_interval:
        if (
            args.conditioning_probe_metadata is None
            or args.conditioning_probe_fingerprints is None
        ):
            raise ValueError(
                "the conditioning probe needs --conditioning-probe-metadata and "
                "--conditioning-probe-fingerprints; a probe drawn from whatever "
                "split happened to be lying around is not comparable between runs"
            )
        if args.conditioning_probe_batch_size < 2:
            # ``mismatched`` conditioning is a roll along the batch.
            raise ValueError("--conditioning-probe-batch-size must be at least 2")
        if args.conditioning_probe_size < 2:
            raise ValueError("--conditioning-probe-size must be at least 2")
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
        if args.selection_metric not in PANEL_SELECTION_METRICS:
            raise ValueError(
                f"--selection-metric {args.selection_metric!r} is not a "
                "molecular-panel metric; expected one of "
                f"{', '.join(PANEL_SELECTION_METRICS)}"
            )
    if args.probe_early_stopping_metric is not None:
        if not args.conditioning_probe_interval:
            raise ValueError(
                "--probe-early-stopping-metric needs "
                "--conditioning-probe-interval: a stopping rule with nothing "
                "to read stops nothing"
            )
        if args.probe_early_stopping_patience <= 0:
            raise ValueError(
                "--probe-early-stopping-patience must be positive; a patience "
                "of 0 records a best probe but never stops, which is the "
                "behaviour --probe-early-stopping-metric exists to replace"
            )
        CheckpointSelector(
            metric=args.probe_early_stopping_metric,
            patience=args.probe_early_stopping_patience,
            min_delta=args.probe_early_stopping_min_delta,
        )
    if args.lr_schedule == "constant":
        if args.derive_learning_rate:
            raise ValueError(
                "--derive-learning-rate has nothing to derive without "
                "--lr-schedule warmup_cosine"
            )
    else:
        if not 0 <= args.lr_warmup_steps < args.max_steps:
            raise ValueError(
                "--lr-warmup-steps must satisfy 0 <= warmup < --max-steps"
            )
        if args.lr_min < 0:
            raise ValueError("--lr-min must be non-negative")
        if not args.derive_learning_rate and args.learning_rate < args.lr_min:
            raise ValueError(
                f"peak --learning-rate {args.learning_rate:g} is below --lr-min "
                f"{args.lr_min:g}"
            )
    if args.time_sampling_eps <= 0 or args.time_sampling_eps >= 1:
        raise ValueError("--time-sampling-eps must be in (0, 1)")


def resolve_learning_rate(args: argparse.Namespace) -> dict[str, object]:
    """Fix the peak rate and record how it was arrived at.

    Returned rather than printed so the manifest and the ClearML config carry
    the derivation: a run whose peak cannot be traced back to a rule is a run
    whose result cannot be attributed to the rule.
    """
    if args.lr_schedule == "constant":
        return {
            "schedule": "constant",
            "peak": args.learning_rate,
            "source": "--learning-rate",
            "released_terminal_learning_rate": RELEASED_TERMINAL_LEARNING_RATE,
            "peak_over_released_terminal": (
                args.learning_rate / RELEASED_TERMINAL_LEARNING_RATE
            ),
            "displacement_bound": args.learning_rate * args.max_steps,
        }
    peak = args.learning_rate
    source = "--learning-rate"
    if args.derive_learning_rate:
        peak = derive_peak_learning_rate(
            total_steps=args.max_steps,
            warmup_steps=args.lr_warmup_steps,
            floor=args.lr_min,
        )
        source = (
            "displacement budget of the released schedule's final decade of "
            "learning rate; see src/marlin/lr_schedule.py"
        )
    return {
        "schedule": "warmup_cosine",
        "peak": peak,
        "source": source,
        "warmup_steps": args.lr_warmup_steps,
        "total_steps": args.max_steps,
        "floor": args.lr_min,
        "released_terminal_learning_rate": RELEASED_TERMINAL_LEARNING_RATE,
        "peak_over_released_terminal": peak / RELEASED_TERMINAL_LEARNING_RATE,
        "displacement_bound": schedule_displacement(
            peak,
            total_steps=args.max_steps,
            warmup_steps=args.lr_warmup_steps,
            floor=args.lr_min,
        ),
    }


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

    learning_rate_plan = resolve_learning_rate(args)
    print(
        "MARLIN adaptation learning rate: "
        f"schedule={learning_rate_plan['schedule']} "
        f"peak={learning_rate_plan['peak']:.6g} "
        f"({learning_rate_plan['peak_over_released_terminal']:.1f}x the "
        "released terminal 5.2697058404552555e-08), displacement bound "
        f"{learning_rate_plan['displacement_bound']:.6g}",
        flush=True,
    )
    module = MarlinLightningModule(
        decoder_config,
        learning_rate=float(learning_rate_plan["peak"]),
        weight_decay=0.0,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_total_steps=args.max_steps,
        lr_min=args.lr_min,
        time_sampling=args.time_sampling,
        time_sampling_eps=args.time_sampling_eps,
        loss_reduction=args.loss_reduction,
        fp32_forward=args.fp32_forward,
        noise_probability=args.noise_probability,
        fingerprint_noise_mode=args.fingerprint_noise_mode,
        noise_min_fraction=args.noise_min_fraction,
        noise_max_fraction=args.noise_max_fraction,
        ema_decay=0.9999,
        metric_interval=25,
        conditioning_only_steps=0,
        cross_attention_only_steps=args.cross_attention_only_steps,
        adapt_fingerprint=True,
        context_corruption_probability=args.context_corruption_probability,
        context_corruption_warmup_steps=args.context_corruption_warmup_steps,
        context_corruption_min_fraction=args.context_corruption_min_fraction,
        context_corruption_max_fraction=args.context_corruption_max_fraction,
        restoration_loss_weight=args.restoration_loss_weight,
    )
    start_report = load_marlin_decoder_weights(
        module.decoder,
        args.checkpoint,
        architecture_upgrade=False,
        use_ema=False,
    )
    module.reset_ema()

    probe_batches = None
    probe_threshold = (
        args.conditioning_probe_threshold
        if args.conditioning_probe_threshold is not None
        else args.validation_fingerprint_threshold
    )
    if args.conditioning_probe_interval:
        # Built before the output directory so a misconfigured probe fails
        # before a run directory and a ClearML task exist.
        probe_batches = probe_batches_from_paths(
            args.conditioning_probe_metadata,
            args.conditioning_probe_fingerprints,
            tokenizer,
            fingerprint_key=args.conditioning_probe_fingerprint_key,
            threshold=probe_threshold,
            max_length=decoder_config.max_length,
            fingerprint_bits=decoder_config.fingerprint_bits,
            size=args.conditioning_probe_size,
            batch_size=args.conditioning_probe_batch_size,
            seed=args.conditioning_probe_seed,
            soft_fingerprint=args.soft_fingerprint,
        )

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
        "learning_rate": float(learning_rate_plan["peak"]),
        "learning_rate_plan": learning_rate_plan,
        "objective": {
            "time_sampling": args.time_sampling,
            "time_sampling_eps": args.time_sampling_eps,
            "loss_reduction": args.loss_reduction,
            "fp32_forward": args.fp32_forward,
        },
        "max_steps": args.max_steps,
        "cross_attention_only_steps": args.cross_attention_only_steps,
        "context_corruption": {
            "probability": args.context_corruption_probability,
            "warmup_steps": args.context_corruption_warmup_steps,
            "min_fraction": args.context_corruption_min_fraction,
            "max_fraction": args.context_corruption_max_fraction,
            "restoration_loss_weight": args.restoration_loss_weight,
            "source": "model runner-up tokens under a fully masked block",
            "targets": "gold",
        },
        "fingerprint_noise": {
            "mode": args.fingerprint_noise_mode,
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
        "instrumentation": {
            "gradient_diagnostics_interval": args.gradient_diagnostics_interval,
            "gradient_clip_val": GRADIENT_CLIP_VAL,
            "conditioning_probe": {
                "interval_steps": args.conditioning_probe_interval,
                "metadata": (
                    str(args.conditioning_probe_metadata)
                    if args.conditioning_probe_metadata
                    else None
                ),
                "fingerprints": (
                    str(args.conditioning_probe_fingerprints)
                    if args.conditioning_probe_fingerprints
                    else None
                ),
                "fingerprint_key": args.conditioning_probe_fingerprint_key,
                "threshold": probe_threshold,
                "size": args.conditioning_probe_size,
                "batch_size": args.conditioning_probe_batch_size,
                "seed": args.conditioning_probe_seed,
                "soft_fingerprint": args.soft_fingerprint,
                "early_stopping": {
                    "metric": args.probe_early_stopping_metric,
                    "patience": args.probe_early_stopping_patience,
                    "min_delta": args.probe_early_stopping_min_delta,
                },
            },
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
        # An early-stopped run ends between two checkpoint intervals, so without
        # this it would leave no weights at all. Only when early stopping can
        # actually fire, so a run without it writes exactly the files it always
        # did.
        save_last=args.probe_early_stopping_metric is not None,
    )
    callbacks = [checkpoint_callback]
    if args.gradient_diagnostics_interval:
        callbacks.extend(
            [
                GradientDiagnostics(
                    interval_steps=args.gradient_diagnostics_interval,
                    clip_value=GRADIENT_CLIP_VAL,
                    clearml_task=clearml_task,
                ),
                EmaDivergence(
                    interval_steps=args.gradient_diagnostics_interval,
                    clearml_task=clearml_task,
                ),
            ]
        )
    if probe_batches is not None:
        probe_selector = (
            CheckpointSelector(
                metric=args.probe_early_stopping_metric,
                patience=args.probe_early_stopping_patience,
                min_delta=args.probe_early_stopping_min_delta,
            )
            if args.probe_early_stopping_metric is not None
            else None
        )
        callbacks.append(
            PeriodicConditioningProbe(
                probe_batches,
                interval_steps=args.conditioning_probe_interval,
                seed=args.conditioning_probe_seed,
                clearml_task=clearml_task,
                selector=probe_selector,
                selection_path=(
                    args.output_dir / "selection" / "probe_selection.json"
                    if probe_selector is not None
                    else None
                ),
            )
        )
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
        gradient_clip_val=GRADIENT_CLIP_VAL,
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
