#!/usr/bin/env python
"""Train the RankLoop spectrum-molecule dual-encoder projection baseline."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from frigid.rankloop_corpus import _git_revision, sha256_file  # noqa: E402
from frigid.rankloop_model import (  # noqa: E402
    DenseRankLoopDualEncoder,
    RankLoopCorpusDataset,
    RankLoopDenseCorpusDataset,
    collate_rankloop_lists,
    compute_rankloop_loss,
    load_molecule_embedding_table,
    load_spectrum_embedding_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RankLoop projection heads over frozen representations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--candidate-corpus", required=True)
    parser.add_argument("--spectrum-metadata", required=True)
    parser.add_argument("--spectrum-embeddings", required=True)
    parser.add_argument("--molecule-metadata")
    parser.add_argument("--molecule-embeddings")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--embedding-dimension", type=int, default=256)
    parser.add_argument("--hidden-dimension", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--symmetric-weight", type=float, default=1.0)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def _move_batch(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def run_epoch(
    model: DenseRankLoopDualEncoder,
    loader: DataLoader,
    *,
    device: torch.device,
    symmetric_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip: float = 1.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {
        "loss": 0.0,
        "list_loss": 0.0,
        "symmetric_loss": 0.0,
        "top1_accuracy": 0.0,
    }
    example_count = 0
    for batch in loader:
        batch = _move_batch(batch, device)
        batch_size = int(batch["spectrum_embeddings"].shape[0])
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits, spectrum_latent, candidate_latent = model(
                batch["spectrum_embeddings"],
                batch["candidate_features"],
            )
            logit_scale = model.logit_scale.clamp(max=math.log(100.0)).exp()
            metrics = compute_rankloop_loss(
                logits,
                spectrum_latent,
                candidate_latent,
                candidate_mask=batch["candidate_mask"],
                positive_mask=batch["positive_mask"],
                query_group_ids=batch["query_group_ids"],
                logit_scale=logit_scale,
                symmetric_weight=symmetric_weight,
            )
            if training:
                metrics["loss"].backward()
                if gradient_clip > 0:
                    clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
        for key in totals:
            totals[key] += float(metrics[key].detach().cpu()) * batch_size
        example_count += batch_size
    if example_count == 0:
        raise ValueError("RankLoop data loader produced no examples.")
    return {key: value / example_count for key, value in totals.items()}


def _checkpoint_payload(
    model: DenseRankLoopDualEncoder,
    *,
    model_config: dict[str, Any],
    epoch: int,
    history: list[dict[str, Any]],
    input_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "architecture": "dense_projection_dual_encoder",
        "model_config": model_config,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "history": history,
        "input_hashes": input_hashes,
    }


def main() -> int:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError(
            "learning_rate must be positive and weight_decay non-negative."
        )
    if args.symmetric_weight < 0:
        raise ValueError("symmetric_weight must be non-negative.")
    seed_everything(args.seed)
    device = resolve_device(args.device)

    candidate_corpus = Path(args.candidate_corpus).expanduser().resolve()
    spectrum_metadata = Path(args.spectrum_metadata).expanduser().resolve()
    spectrum_embeddings = Path(args.spectrum_embeddings).expanduser().resolve()
    if bool(args.molecule_metadata) != bool(args.molecule_embeddings):
        raise ValueError(
            "molecule_metadata and molecule_embeddings must be provided together."
        )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_table = load_spectrum_embedding_table(
        spectrum_metadata,
        spectrum_embeddings,
    )
    if args.molecule_metadata:
        molecule_metadata = Path(args.molecule_metadata).expanduser().resolve()
        molecule_embeddings = Path(args.molecule_embeddings).expanduser().resolve()
        molecule_table = load_molecule_embedding_table(
            molecule_metadata,
            molecule_embeddings,
        )
        train_dataset = RankLoopDenseCorpusDataset(
            candidate_corpus,
            embedding_table,
            molecule_table,
            partition="train",
        )
        development_dataset = RankLoopDenseCorpusDataset(
            candidate_corpus,
            embedding_table,
            molecule_table,
            partition="development",
        )
        molecule_mode = "precomputed"
        molecule_dimension = molecule_table.dimension
    else:
        molecule_metadata = None
        molecule_embeddings = None
        train_dataset = RankLoopCorpusDataset(
            candidate_corpus,
            embedding_table,
            partition="train",
            fingerprint_bits=args.fingerprint_bits,
            fingerprint_radius=args.fingerprint_radius,
        )
        development_dataset = RankLoopCorpusDataset(
            candidate_corpus,
            embedding_table,
            partition="development",
            fingerprint_bits=args.fingerprint_bits,
            fingerprint_radius=args.fingerprint_radius,
        )
        molecule_mode = "morgan"
        molecule_dimension = args.fingerprint_bits
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_rankloop_lists,
    )
    development_loader = DataLoader(
        development_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_rankloop_lists,
    )
    model_config = {
        "spectrum_dimension": embedding_table.dimension,
        "molecule_dimension": molecule_dimension,
        "embedding_dimension": args.embedding_dimension,
        "hidden_dimension": args.hidden_dimension,
        "dropout": args.dropout,
        "temperature": args.temperature,
    }
    model = DenseRankLoopDualEncoder(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    input_hashes = {
        "candidate_corpus": sha256_file(candidate_corpus),
        "spectrum_metadata": sha256_file(spectrum_metadata),
        "spectrum_embeddings": sha256_file(spectrum_embeddings),
    }
    if molecule_metadata is not None and molecule_embeddings is not None:
        input_hashes.update(
            {
                "molecule_metadata": sha256_file(molecule_metadata),
                "molecule_embeddings": sha256_file(molecule_embeddings),
            }
        )

    history: list[dict[str, Any]] = []
    best_key = (-math.inf, -math.inf)
    best_development_loss = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = output_dir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device=device,
            symmetric_weight=args.symmetric_weight,
            optimizer=optimizer,
            gradient_clip=args.gradient_clip,
        )
        with torch.no_grad():
            development_metrics = run_epoch(
                model,
                development_loader,
                device=device,
                symmetric_weight=args.symmetric_weight,
            )
        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "development": development_metrics,
            "logit_scale": float(
                model.logit_scale.clamp(max=math.log(100.0)).exp().detach().cpu()
            ),
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, sort_keys=True), flush=True)
        score = (
            development_metrics["top1_accuracy"],
            -development_metrics["loss"],
        )
        if score > best_key:
            best_key = score
            best_development_loss = development_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                _checkpoint_payload(
                    model,
                    model_config=model_config,
                    epoch=epoch,
                    history=history,
                    input_hashes=input_hashes,
                ),
                best_path,
            )
        else:
            epochs_without_improvement += 1
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            break

    last_path = output_dir / "last.pt"
    torch.save(
        _checkpoint_payload(
            model,
            model_config=model_config,
            epoch=history[-1]["epoch"],
            history=history,
            input_hashes=input_hashes,
        ),
        last_path,
    )
    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    revision, dirty = _git_revision(PROJECT_ROOT)
    manifest = {
        "schema_version": 1,
        "state": "completed",
        "architecture": "dense_projection_dual_encoder",
        "molecule_mode": molecule_mode,
        "evidence_class": (
            "prepared_model_not_quality_evidence"
            if molecule_mode == "precomputed"
            else "infrastructure_baseline_not_quality_candidate"
        ),
        "repo": {"commit": revision, "dirty": dirty},
        "device": str(device),
        "parameters": vars(args),
        "dataset": {
            "train_queries": len(train_dataset),
            "development_queries": len(development_dataset),
            "spectrum_dimension": embedding_table.dimension,
        },
        "inputs": input_hashes,
        "selection": {
            "best_epoch": best_epoch,
            "development_top1_accuracy": best_key[0],
            "development_loss": best_development_loss,
        },
        "outputs": {
            "best_checkpoint": str(best_path),
            "best_checkpoint_sha256": sha256_file(best_path),
            "last_checkpoint": str(last_path),
            "last_checkpoint_sha256": sha256_file(last_path),
            "history": str(output_dir / "history.json"),
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest["selection"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
