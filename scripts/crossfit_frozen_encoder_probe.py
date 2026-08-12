#!/usr/bin/env python
"""Predict a split's fingerprints out of fold, so no row is scored by a probe
that was fitted on it.

`scripts/train_frozen_encoder_probe.py` fits one probe on the whole train split
and then applies it to that same split to produce
`train/dreams_predictions.npz`. Those predictions are therefore **in sample**:
median Tanimoto to the true Morgan fingerprint is 0.475 on train against 0.282
on the held-out validation split, so a decoder adapted on them is taught a
conditioning distribution 1.7x cleaner than the one it meets at inference.

This script removes that gap by k-fold cross-fitting. Folds are whole
connectivity blocks, assigned by the same stable hash
`frigid.frozen_probe.deterministic_group_holdout` uses, so a structure never
appears in both the fit and the prediction of the same fold. Inside each fold's
fit rows a further group holdout selects the epoch, exactly as the single-probe
recipe does, and the fold probe is then refit from scratch on all of its fit
rows for that many epochs.

Usage:
  PYTHONPATH=src python scripts/crossfit_frozen_encoder_probe.py \
      --bundle runs/dreams/train.npz \
      --output runs/dreams/probe/train_predictions_oof.npz \
      --folds 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from frigid.encoder_benchmark import sha256_file  # noqa: E402
from frigid.frozen_probe import (  # noqa: E402
    deterministic_group_holdout,
    global_positive_weight,
    load_embedding_bundle,
)
from train_frozen_encoder_probe import (  # noqa: E402
    DEFAULT_SELECTION_THRESHOLD_GRID,
    FrozenFingerprintProbe,
    _internal_validation_metrics,
    _make_loader,
    _parse_threshold_grid,
    _resolve_device,
    _train_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-fit a frozen-encoder Morgan probe and predict out of fold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fingerprint-bits", type=int, default=4096)
    parser.add_argument("--internal-validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--minimum-tanimoto-improvement", type=float, default=1e-5)
    parser.add_argument(
        "--selection-threshold-grid", default=DEFAULT_SELECTION_THRESHOLD_GRID
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def assign_folds(structure_ids: np.ndarray, *, folds: int, seed: int) -> np.ndarray:
    """Map every row to a fold, keeping a connectivity block undivided."""

    if folds < 2:
        raise ValueError("--folds must be at least 2")
    unique = sorted(set(structure_ids.tolist()))
    if len(unique) < folds:
        raise ValueError("fewer connectivity blocks than folds")
    ranked = sorted(
        unique,
        key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).digest(),
    )
    fold_of_structure = {
        structure: index % folds for index, structure in enumerate(ranked)
    }
    return np.asarray(
        [fold_of_structure[value] for value in structure_ids.tolist()], dtype=np.int64
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fit_probe(
    dataset: TensorDataset,
    targets: np.ndarray,
    structure_ids: np.ndarray,
    fit_rows: np.ndarray,
    *,
    args: argparse.Namespace,
    thresholds: list[float],
    device: torch.device,
    input_dim: int,
    seed: int,
) -> tuple[FrozenFingerprintProbe, dict[str, object]]:
    """Select the epoch on a held-out group split, then refit on every fit row."""

    pin_memory = device.type == "cuda"
    inner_fit, inner_holdout = deterministic_group_holdout(
        structure_ids[fit_rows],
        validation_fraction=args.internal_validation_fraction,
        seed=seed,
    )
    inner_fit_rows = fit_rows[inner_fit]
    inner_holdout_rows = fit_rows[inner_holdout]

    _seed_everything(seed)
    model = FrozenFingerprintProbe(
        input_dim=input_dim, fingerprint_bits=args.fingerprint_bits
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [global_positive_weight(targets, inner_fit_rows)],
            dtype=torch.float32,
            device=device,
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    fit_loader = _make_loader(
        dataset,
        inner_fit_rows,
        batch_size=args.batch_size,
        shuffle=True,
        seed=seed,
        pin_memory=pin_memory,
    )
    holdout_loader = _make_loader(
        dataset,
        inner_holdout_rows,
        batch_size=args.batch_size,
        shuffle=False,
        seed=seed,
        pin_memory=pin_memory,
    )
    best_metric = -1.0
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        _train_epoch(model, fit_loader, criterion, optimizer, device)
        _, tanimoto, _ = _internal_validation_metrics(
            model, holdout_loader, criterion, device, thresholds
        )
        if tanimoto > best_metric + args.minimum_tanimoto_improvement:
            best_metric = tanimoto
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    _seed_everything(seed)
    model = FrozenFingerprintProbe(
        input_dim=input_dim, fingerprint_bits=args.fingerprint_bits
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [global_positive_weight(targets, fit_rows)],
            dtype=torch.float32,
            device=device,
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    refit_loader = _make_loader(
        dataset,
        fit_rows,
        batch_size=args.batch_size,
        shuffle=True,
        seed=seed,
        pin_memory=pin_memory,
    )
    for _ in range(max(best_epoch, 1)):
        _train_epoch(model, refit_loader, criterion, optimizer, device)
    return model, {
        "selected_epoch": best_epoch,
        "selection_internal_holdout_tanimoto": best_metric,
        "fit_rows": int(len(fit_rows)),
        "inner_fit_rows": int(len(inner_fit_rows)),
        "inner_holdout_rows": int(len(inner_holdout_rows)),
    }


def _predict_rows(
    model: FrozenFingerprintProbe,
    dataset: TensorDataset,
    rows: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = _make_loader(
        dataset,
        rows,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    chunks = []
    with torch.inference_mode():
        for embeddings, _ in loader:
            embeddings = embeddings.to(device=device, dtype=torch.float32)
            chunks.append(torch.sigmoid(model(embeddings)).cpu().numpy())
    return np.vstack(chunks)


def median_tanimoto(
    probabilities: np.ndarray, targets: np.ndarray, threshold: float
) -> float:
    predicted = probabilities >= threshold
    truth = targets > 0.5
    intersection = np.logical_and(predicted, truth).sum(axis=1)
    union = np.logical_or(predicted, truth).sum(axis=1)
    tanimoto = np.where(union > 0, intersection / np.maximum(union, 1), 0.0)
    return float(np.median(tanimoto))


def main() -> int:
    args = parse_args()
    thresholds = _parse_threshold_grid(args.selection_threshold_grid)
    bundle = load_embedding_bundle(args.bundle, fingerprint_bits=args.fingerprint_bits)
    device = _resolve_device(args.device)
    dataset = TensorDataset(
        torch.from_numpy(bundle.embeddings), torch.from_numpy(bundle.targets)
    )
    fold_of_row = assign_folds(
        bundle.structure_ids, folds=args.folds, seed=args.seed
    )
    predictions = np.zeros(
        (len(bundle.spectrum_ids), args.fingerprint_bits), dtype=np.float32
    )
    covered = np.zeros(len(bundle.spectrum_ids), dtype=bool)
    fold_reports = []
    started = time.perf_counter()
    for fold in range(args.folds):
        held_out = np.flatnonzero(fold_of_row == fold)
        fit_rows = np.flatnonzero(fold_of_row != fold)
        if len(held_out) == 0:
            raise ValueError(f"fold {fold} is empty")
        overlap = set(bundle.structure_ids[held_out].tolist()) & set(
            bundle.structure_ids[fit_rows].tolist()
        )
        if overlap:
            raise ValueError(f"fold {fold} leaks structures: {sorted(overlap)[:5]}")
        fold_started = time.perf_counter()
        model, report = _fit_probe(
            dataset,
            bundle.targets,
            bundle.structure_ids,
            fit_rows,
            args=args,
            thresholds=thresholds,
            device=device,
            input_dim=bundle.embeddings.shape[1],
            seed=args.seed + fold,
        )
        predictions[held_out] = _predict_rows(
            model, dataset, held_out, batch_size=args.batch_size, device=device
        )
        covered[held_out] = True
        report.update(
            {
                "fold": fold,
                "held_out_rows": int(len(held_out)),
                "held_out_structures": int(
                    len(set(bundle.structure_ids[held_out].tolist()))
                ),
                "held_out_median_tanimoto_0.95": median_tanimoto(
                    predictions[held_out], bundle.targets[held_out], 0.95
                ),
                "seconds": time.perf_counter() - fold_started,
            }
        )
        fold_reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)
    if not covered.all():
        raise RuntimeError("some rows were never predicted out of fold")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    inference_seconds = (
        bundle.inference_seconds
        if bundle.inference_seconds is not None
        else np.zeros(len(bundle.spectrum_ids), dtype=np.float64)
    )
    np.savez_compressed(
        args.output,
        probs=predictions,
        spectrum_ids=bundle.spectrum_ids,
        inference_seconds=inference_seconds,
    )
    summary = {
        "bundle": str(Path(args.bundle).resolve()),
        "bundle_sha256": sha256_file(args.bundle),
        "output_sha256": sha256_file(args.output),
        "rows": int(len(bundle.spectrum_ids)),
        "structures": int(len(set(bundle.structure_ids.tolist()))),
        "folds": args.folds,
        "seed": args.seed,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "out_of_fold_median_tanimoto_0.95": median_tanimoto(
            predictions, bundle.targets, 0.95
        ),
        "out_of_fold_median_tanimoto_0.90": median_tanimoto(
            predictions, bundle.targets, 0.90
        ),
        "out_of_fold_mean_active_bits_0.95": float(
            (predictions >= 0.95).sum(axis=1).mean()
        ),
        "fold_reports": fold_reports,
    }
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"summary": str(summary_path), **{
        key: summary[key]
        for key in (
            "out_of_fold_median_tanimoto_0.95",
            "out_of_fold_mean_active_bits_0.95",
            "elapsed_seconds",
        )
    }}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
