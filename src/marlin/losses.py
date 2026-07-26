"""Shared loss-weighting utilities for MARLIN objectives."""

from __future__ import annotations

import math

import torch


def balanced_token_target_weights(
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    vocab_size: int,
    eos_token_id: int,
    alpha: float = 0.0,
    weight_max: float = 20.0,
) -> torch.Tensor:
    """Return strict inverse-frequency content-token weights.

    Frequencies are computed over all valid non-EOS targets in the batch.
    EOS deliberately keeps unit weight, matching the original MARLIN
    reconstruction objective.
    """

    if targets.shape != valid_mask.shape or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean with the target shape")
    if targets.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError("targets must contain integer token IDs")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise ValueError("vocab_size must be a positive integer")
    if vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer")
    if isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int):
        raise ValueError("eos_token_id must be an integer in the vocabulary")
    if not 0 <= eos_token_id < vocab_size:
        raise ValueError("eos_token_id must be an integer in the vocabulary")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("balanced_token_loss_alpha must be finite and non-negative")
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("balanced_token_loss_alpha must be finite and non-negative")
    if isinstance(weight_max, bool) or not isinstance(weight_max, (int, float)):
        raise ValueError("token_loss_weight_max must be finite and at least 1")
    weight_max = float(weight_max)
    if not math.isfinite(weight_max) or weight_max < 1:
        raise ValueError("token_loss_weight_max must be finite and at least 1")
    if valid_mask.any():
        valid_targets = targets[valid_mask]
        if ((valid_targets < 0) | (valid_targets >= vocab_size)).any():
            raise ValueError("valid target token ID is outside the vocabulary")

    target_weights = torch.ones_like(targets, dtype=torch.float32)
    if alpha == 0:
        return target_weights

    content_targets = valid_mask & targets.ne(eos_token_id)
    selected_targets = targets[content_targets]
    if selected_targets.numel() == 0:
        return target_weights
    counts = torch.bincount(
        selected_targets.to(dtype=torch.long),
        minlength=vocab_size,
    ).to(dtype=target_weights.dtype)
    positive = counts.gt(0)
    mean_count = counts[positive].mean()
    vocabulary_weights = torch.ones_like(counts)
    vocabulary_weights[positive] = (
        mean_count / counts[positive]
    ).pow(alpha)
    vocabulary_weights.clamp_(max=weight_max)
    target_weights[content_targets] = vocabulary_weights[
        targets[content_targets].to(dtype=torch.long)
    ]
    return target_weights
