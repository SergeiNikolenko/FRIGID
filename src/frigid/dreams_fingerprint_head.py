"""Utilities for a DreaMS-embedding to Morgan-fingerprint head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


class FingerprintHead(nn.Module):
    """Small supervised head mapping spectrum embeddings to fingerprint logits."""

    def __init__(
        self,
        input_dim: int,
        fingerprint_bits: int = 4096,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fingerprint_bits),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)


class FingerprintEmbeddingDataset(Dataset):
    """Dataset backed by embedding and target-fingerprint arrays."""

    def __init__(self, embeddings: np.ndarray, fingerprints: np.ndarray):
        if embeddings.ndim != 2:
            raise ValueError(f"Expected embeddings to be 2D, got {embeddings.shape}")
        if fingerprints.ndim != 2:
            raise ValueError(f"Expected fingerprints to be 2D, got {fingerprints.shape}")
        if len(embeddings) != len(fingerprints):
            raise ValueError(
                f"Embedding rows ({len(embeddings)}) do not match fingerprint rows "
                f"({len(fingerprints)})"
            )
        self.embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        self.fingerprints = torch.as_tensor(fingerprints, dtype=torch.float32)

    def __len__(self) -> int:
        return self.embeddings.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[index], self.fingerprints[index]


@dataclass
class FingerprintMetrics:
    mean_tanimoto: float
    median_tanimoto: float
    bit_accuracy: float
    bit_precision: float
    bit_recall: float
    bit_f1: float
    mean_active_bits: float
    mean_target_active_bits: float


def tanimoto_per_row(pred_binary: np.ndarray, target_binary: np.ndarray) -> np.ndarray:
    """Compute binary Tanimoto similarity row-wise."""
    intersection = np.minimum(pred_binary, target_binary).sum(axis=1)
    union = np.maximum(pred_binary, target_binary).sum(axis=1)
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float64),
        where=union > 0,
    )


def compute_fingerprint_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> Tuple[FingerprintMetrics, np.ndarray]:
    """Compute fingerprint-head metrics from probabilities and binary targets."""
    probs = np.asarray(probs, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    pred = (probs >= threshold).astype(np.float32)
    target = (targets > 0.5).astype(np.float32)

    tanimoto = tanimoto_per_row(pred, target)
    tp = float(np.logical_and(pred == 1, target == 1).sum())
    fp = float(np.logical_and(pred == 1, target == 0).sum())
    fn = float(np.logical_and(pred == 0, target == 1).sum())
    bit_accuracy = float((pred == target).mean())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    metrics = FingerprintMetrics(
        mean_tanimoto=float(tanimoto.mean()) if len(tanimoto) else 0.0,
        median_tanimoto=float(np.median(tanimoto)) if len(tanimoto) else 0.0,
        bit_accuracy=bit_accuracy,
        bit_precision=precision,
        bit_recall=recall,
        bit_f1=f1,
        mean_active_bits=float(pred.sum(axis=1).mean()) if len(pred) else 0.0,
        mean_target_active_bits=float(target.sum(axis=1).mean()) if len(target) else 0.0,
    )
    return metrics, tanimoto


def load_npz_array(path: str, key: str) -> np.ndarray:
    arrays = np.load(path)
    if key not in arrays:
        raise ValueError(f"Array {key!r} not found in {path}; available: {arrays.files}")
    return arrays[key]


def checkpoint_payload(
    model: FingerprintHead,
    *,
    input_dim: int,
    fingerprint_bits: int,
    hidden_dim: int,
    dropout: float,
    threshold: float,
    metrics: Dict,
) -> Dict:
    return {
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "fingerprint_bits": fingerprint_bits,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "threshold": threshold,
        "metrics": metrics,
    }
