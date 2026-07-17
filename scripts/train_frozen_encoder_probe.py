#!/usr/bin/env python
"""Train the shared LayerNorm-to-linear probe on frozen spectrum embeddings."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from frigid.encoder_benchmark import sha256_file  # noqa: E402
from frigid.frozen_probe import (  # noqa: E402
    EmbeddingBundle,
    deterministic_group_holdout,
    global_positive_weight,
    load_embedding_bundle,
)


DEFAULT_SELECTION_THRESHOLD_GRID = (
    "0.005,0.01,0.02,0.03,0.04,0.05,0.075,0.1,0.125,0.15,0.2,0.25,"
    "0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.85,0.9,0.925,0.95,0.975,0.99"
)


class FrozenFingerprintProbe(nn.Module):
    """The architecture shared by every frozen dense encoder candidate."""

    def __init__(self, input_dim: int, fingerprint_bits: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.output = nn.Linear(input_dim, fingerprint_bits)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.output(self.layer_norm(embeddings))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frozen-encoder Morgan-4096 probe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-npz", required=True)
    parser.add_argument("--validation-npz", required=True)
    parser.add_argument("--output-dir", required=True)
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
    parser.add_argument(
        "--drop-validation-overlap",
        action="store_true",
        help="Remove validation connectivity blocks from training before fitting.",
    )
    return parser.parse_args()


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(value)


def _make_loader(
    dataset: TensorDataset,
    indexes: np.ndarray | None,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    selected = dataset if indexes is None else Subset(dataset, indexes.tolist())
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        selected,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
        generator=generator,
    )


def _parse_threshold_grid(value: str) -> list[float]:
    thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not thresholds or any(
        not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0
        for threshold in thresholds
    ):
        raise ValueError(f"Invalid selection threshold grid: {value!r}")
    return thresholds


def _internal_validation_metrics(
    model: FrozenFingerprintProbe,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    thresholds: list[float],
) -> tuple[float, float, float]:
    model.eval()
    loss_sum = 0.0
    tanimoto_sums = np.zeros(len(thresholds), dtype=np.float64)
    rows = 0
    with torch.inference_mode():
        for embeddings, targets in loader:
            embeddings = embeddings.to(device=device, dtype=torch.float32, non_blocking=True)
            targets = targets.to(device=device, dtype=torch.float32, non_blocking=True)
            logits = model(embeddings)
            probabilities = torch.sigmoid(logits)
            loss = criterion(logits, targets)
            loss_sum += float(loss.item()) * len(embeddings)
            target_binary = targets > 0.5
            for index, threshold in enumerate(thresholds):
                prediction_binary = probabilities >= threshold
                intersection = torch.logical_and(
                    prediction_binary, target_binary
                ).sum(dim=1)
                union = torch.logical_or(prediction_binary, target_binary).sum(dim=1)
                tanimoto = torch.where(
                    union > 0,
                    intersection.to(torch.float32) / union.to(torch.float32),
                    torch.zeros_like(union, dtype=torch.float32),
                )
                tanimoto_sums[index] += float(tanimoto.sum().item())
            rows += len(embeddings)
    mean_tanimotos = tanimoto_sums / rows
    best_index = max(
        range(len(thresholds)),
        key=lambda index: (mean_tanimotos[index], -thresholds[index]),
    )
    return (
        loss_sum / rows,
        float(mean_tanimotos[best_index]),
        thresholds[best_index],
    )


def _train_epoch(
    model: FrozenFingerprintProbe,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_sum = 0.0
    rows = 0
    for embeddings, targets in loader:
        embeddings = embeddings.to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        targets = targets.to(device=device, dtype=torch.float32, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(embeddings), targets)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * len(embeddings)
        rows += len(embeddings)
    return loss_sum / rows


def _predict(
    model: FrozenFingerprintProbe,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    model.eval()
    predictions = []
    elapsed = 0.0
    with torch.inference_mode():
        for embeddings, _ in loader:
            embeddings = embeddings.to(device=device, dtype=torch.float32, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            probabilities = torch.sigmoid(model(embeddings))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - started
            predictions.append(probabilities.cpu().numpy())
    return np.vstack(predictions), elapsed


def main() -> int:
    args = parse_args()
    if args.fingerprint_bits <= 0 or args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("fingerprint-bits, batch-size, and epochs must be positive")
    if args.patience <= 0:
        raise ValueError("patience must be positive")
    selection_thresholds = _parse_threshold_grid(args.selection_threshold_grid)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train = load_embedding_bundle(args.train_npz, fingerprint_bits=args.fingerprint_bits)
    validation = load_embedding_bundle(
        args.validation_npz, fingerprint_bits=args.fingerprint_bits
    )
    if train.embeddings.shape[1] != validation.embeddings.shape[1]:
        raise ValueError(
            "Train and validation embedding widths differ: "
            f"{train.embeddings.shape[1]} versus {validation.embeddings.shape[1]}"
        )
    overlap = set(train.structure_ids.tolist()) & set(validation.structure_ids.tolist())
    dropped_overlap_rows = 0
    if overlap and args.drop_validation_overlap:
        keep = np.asarray(
            [structure_id not in overlap for structure_id in train.structure_ids],
            dtype=bool,
        )
        dropped_overlap_rows = int((~keep).sum())
        train = EmbeddingBundle(
            embeddings=train.embeddings[keep],
            targets=train.targets[keep],
            spectrum_ids=train.spectrum_ids[keep],
            structure_ids=train.structure_ids[keep],
            inference_seconds=(
                train.inference_seconds[keep]
                if train.inference_seconds is not None
                else None
            ),
        )
        overlap = set(train.structure_ids.tolist()) & set(validation.structure_ids.tolist())
    if overlap:
        raise ValueError(
            "Train/validation structure overlap is forbidden; examples: "
            f"{sorted(overlap)[:5]}"
        )

    train_indexes, internal_validation_indexes = deterministic_group_holdout(
        train.structure_ids,
        validation_fraction=args.internal_validation_fraction,
        seed=args.seed,
    )
    positive_weight = global_positive_weight(train.targets, train_indexes)
    device = _resolve_device(args.device)
    pin_memory = device.type == "cuda"

    train_dataset = TensorDataset(
        torch.from_numpy(train.embeddings), torch.from_numpy(train.targets)
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(validation.embeddings), torch.from_numpy(validation.targets)
    )
    train_loader = _make_loader(
        train_dataset,
        train_indexes,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        pin_memory=pin_memory,
    )
    internal_validation_loader = _make_loader(
        train_dataset,
        internal_validation_indexes,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        pin_memory=pin_memory,
    )
    prediction_loader = _make_loader(
        validation_dataset,
        None,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        pin_memory=pin_memory,
    )

    model = FrozenFingerprintProbe(
        input_dim=train.embeddings.shape[1], fingerprint_bits=args.fingerprint_bits
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    history = []
    best_metric = -1.0
    best_loss = float("inf")
    best_threshold = 0.0
    best_epoch = 0
    stale_epochs = 0
    selection_checkpoint_path = output_dir / "selection_probe.pt"
    selection_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)

        (
            internal_validation_loss,
            internal_validation_tanimoto,
            internal_validation_threshold,
        ) = _internal_validation_metrics(
            model,
            internal_validation_loader,
            criterion,
            device,
            selection_thresholds,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "internal_validation_loss": internal_validation_loss,
            "internal_validation_tanimoto": internal_validation_tanimoto,
            "internal_validation_threshold": internal_validation_threshold,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if (
            internal_validation_tanimoto
            > best_metric + args.minimum_tanimoto_improvement
        ):
            best_metric = internal_validation_tanimoto
            best_loss = internal_validation_loss
            best_threshold = internal_validation_threshold
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "input_dim": train.embeddings.shape[1],
                    "fingerprint_bits": args.fingerprint_bits,
                    "epoch": epoch,
                    "internal_validation_loss": best_loss,
                    "internal_validation_tanimoto": best_metric,
                    "internal_validation_threshold": best_threshold,
                },
                selection_checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break
    selection_training_seconds = time.perf_counter() - selection_started

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    all_train_indexes = np.arange(len(train.spectrum_ids), dtype=np.int64)
    final_positive_weight = global_positive_weight(train.targets, all_train_indexes)
    final_train_loader = _make_loader(
        train_dataset,
        None,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        pin_memory=pin_memory,
    )
    model = FrozenFingerprintProbe(
        input_dim=train.embeddings.shape[1], fingerprint_bits=args.fingerprint_bits
    ).to(device)
    final_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [final_positive_weight], dtype=torch.float32, device=device
        )
    )
    final_optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    final_history = []
    final_training_started = time.perf_counter()
    for epoch in range(1, best_epoch + 1):
        train_loss = _train_epoch(
            model, final_train_loader, final_criterion, final_optimizer, device
        )
        final_row = {"epoch": epoch, "train_loss": train_loss}
        final_history.append(final_row)
        print(json.dumps({"final_retrain": final_row}, sort_keys=True), flush=True)
    final_training_seconds = time.perf_counter() - final_training_started
    checkpoint_path = output_dir / "frozen_probe.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": train.embeddings.shape[1],
            "fingerprint_bits": args.fingerprint_bits,
            "selected_epoch": best_epoch,
            "selection_internal_validation_tanimoto": best_metric,
            "selection_internal_validation_threshold": best_threshold,
        },
        checkpoint_path,
    )
    probabilities, head_inference_seconds = _predict(model, prediction_loader, device)
    per_row_seconds = np.full(
        len(validation.spectrum_ids),
        head_inference_seconds / len(validation.spectrum_ids),
        dtype=np.float64,
    )
    if validation.inference_seconds is not None:
        per_row_seconds += validation.inference_seconds

    predictions_path = output_dir / "predictions.npz"
    np.savez_compressed(
        predictions_path,
        probs=probabilities.astype(np.float32, copy=False),
        spectrum_ids=validation.spectrum_ids,
        inference_seconds=per_row_seconds,
    )
    training_identifiers_path = output_dir / "training_inchikeys.txt"
    training_identifiers_path.write_text(
        "\n".join(sorted(set(train.structure_ids.tolist()))) + "\n"
    )
    summary = {
        "architecture": (
            f"LayerNorm(input_dim) -> Linear(input_dim, {args.fingerprint_bits})"
        ),
        "train_npz": os.path.abspath(args.train_npz),
        "train_npz_sha256": sha256_file(args.train_npz),
        "validation_npz": os.path.abspath(args.validation_npz),
        "validation_npz_sha256": sha256_file(args.validation_npz),
        "predictions_sha256": sha256_file(predictions_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "selection_checkpoint_sha256": sha256_file(selection_checkpoint_path),
        "train_rows": len(train.spectrum_ids),
        "probe_train_rows": len(train_indexes),
        "internal_validation_rows": len(internal_validation_indexes),
        "final_train_rows": len(train.spectrum_ids),
        "validation_rows": len(validation.spectrum_ids),
        "train_structures": len(set(train.structure_ids.tolist())),
        "validation_structures": len(set(validation.structure_ids.tolist())),
        "train_validation_structure_overlap": 0,
        "dropped_validation_overlap_rows": dropped_overlap_rows,
        "input_dim": train.embeddings.shape[1],
        "fingerprint_bits": args.fingerprint_bits,
        "positive_weight": final_positive_weight,
        "selection_positive_weight": positive_weight,
        "device": str(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "best_epoch": best_epoch,
        "best_internal_validation_loss": best_loss,
        "best_internal_validation_tanimoto": best_metric,
        "best_internal_validation_threshold": best_threshold,
        "selection_threshold_grid": selection_thresholds,
        "selection_training_seconds": selection_training_seconds,
        "final_training_seconds": final_training_seconds,
        "training_seconds": selection_training_seconds + final_training_seconds,
        "head_inference_seconds": head_inference_seconds,
        "head_inference_timing_scope": (
            "synchronized LayerNorm, linear projection, and sigmoid forward only"
        ),
        "combined_inference_timing_scope": (
            "encoder bundle per-row inference plus probe forward; preprocessing and "
            "host/device transfers are exporter-dependent and excluded when supported"
        ),
        "history": history,
        "final_history": final_history,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
