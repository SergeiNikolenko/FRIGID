"""Bounded causal diagnostics for MARLIN conditioning separability."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from marlin.model import MarlinDecoder


HIDDEN_STAGE_NAMES = (
    "first_self_attention_output",
    "first_cross_attention_output",
    "last_decoder_layer_pre_head",
    "prediction_norm_post_head",
)
ALL_STAGE_NAMES = (*HIDDEN_STAGE_NAMES, "raw_logits")


def _tensor_output(output: Any, stage: str) -> torch.Tensor:
    tensor = output[0] if isinstance(output, tuple) else output
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{stage} hook did not return a tensor")
    return tensor


@torch.inference_mode()
def capture_action_stages(
    model: MarlinDecoder,
    input_ids: torch.Tensor,
    precursor_mass: torch.Tensor,
    fingerprint: torch.Tensor,
    action_positions: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Capture decoder stages at each action's noisy-stream position.

    Results are detached FP32 CPU tensors with one row per input action. The
    model is expected to be in evaluation mode and receives no autocasting.
    """

    if not model.layers:
        raise ValueError("hidden separability requires at least one decoder layer")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, length]")
    batch_size, sequence_length = input_ids.shape
    if action_positions.shape != (batch_size,):
        raise ValueError("action_positions must have shape [batch]")
    if action_positions.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError("action_positions must use an integer dtype")
    if bool(((action_positions < 0) | (action_positions >= sequence_length)).any()):
        raise ValueError("action_positions are outside the input sequence")
    if precursor_mass.shape != (batch_size,):
        raise ValueError("precursor_mass must have shape [batch]")
    if fingerprint.shape != (batch_size, model.config.fingerprint_bits):
        raise ValueError(
            "fingerprint must have shape "
            f"[batch, {model.config.fingerprint_bits}]"
        )

    captured: dict[str, torch.Tensor] = {}

    def save(stage: str):
        def hook(_module, _inputs, output) -> None:
            if stage in captured:
                raise RuntimeError(f"{stage} hook fired more than once")
            captured[stage] = _tensor_output(output, stage)

        return hook

    handles = [
        model.layers[0].self_attention.register_forward_hook(
            save("first_self_attention_output")
        ),
        model.layers[0].cross_attention.register_forward_hook(
            save("first_cross_attention_output")
        ),
        model.layers[-1].register_forward_hook(
            save("last_decoder_layer_pre_head")
        ),
        model.prediction_norm.register_forward_hook(
            save("prediction_norm_post_head")
        ),
    ]
    try:
        logits = model.sampling_logits(input_ids, precursor_mass, fingerprint)
    finally:
        for handle in handles:
            handle.remove()

    missing = set(HIDDEN_STAGE_NAMES).difference(captured)
    if missing:
        raise RuntimeError(f"missing hidden-stage hooks: {sorted(missing)}")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError(
            "sampling logits have unexpected leading shape: "
            f"{tuple(logits.shape)}"
        )

    batch_indices = torch.arange(batch_size, device=input_ids.device)
    action_positions = action_positions.to(device=input_ids.device, dtype=torch.long)
    noisy_positions = action_positions + sequence_length
    selected: dict[str, torch.Tensor] = {}
    for stage in HIDDEN_STAGE_NAMES:
        tensor = captured[stage]
        if tensor.ndim != 3 or tensor.shape[:2] != (
            batch_size,
            sequence_length * 2,
        ):
            raise ValueError(
                f"{stage} has unexpected shape {tuple(tensor.shape)}"
            )
        selected[stage] = (
            tensor[batch_indices, noisy_positions]
            .detach()
            .to(device="cpu", dtype=torch.float32)
        )
    selected["raw_logits"] = (
        logits[batch_indices, action_positions]
        .detach()
        .to(device="cpu", dtype=torch.float32)
    )
    return selected


def normalized_rms_per_row(
    reference: torch.Tensor,
    intervention: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Return RMS(reference-intervention) divided by RMS(reference)."""

    if reference.shape != intervention.shape or reference.ndim != 2:
        raise ValueError("normalized RMS inputs must have equal [rows, features] shape")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    reference = reference.float()
    intervention = intervention.float()
    numerator = (reference - intervention).square().mean(dim=-1).sqrt()
    denominator = reference.square().mean(dim=-1).sqrt()
    return numerator / denominator.clamp_min(epsilon)


def symmetric_normalized_rms(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> float:
    """Return a direction-independent normalized RMS between two vectors."""

    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("symmetric normalized RMS inputs must be equal vectors")
    numerator = (first.float() - second.float()).square().mean().sqrt()
    denominator = (
        0.5
        * (
            first.float().square().mean()
            + second.float().square().mean()
        )
    ).sqrt()
    return float((numerator / denominator.clamp_min(epsilon)).item())


def summarize_values(values: torch.Tensor) -> dict[str, float | int]:
    """Return deterministic compact statistics for a non-empty finite vector."""

    values = values.detach().float().reshape(-1)
    if values.numel() == 0:
        raise ValueError("cannot summarize an empty vector")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("cannot summarize non-finite values")
    return {
        "count": int(values.numel()),
        "median": float(torch.quantile(values, 0.5).item()),
        "p90": float(torch.quantile(values, 0.9).item()),
        "minimum": float(values.min().item()),
        "maximum": float(values.max().item()),
    }


def summarize_intervention(
    correct: Mapping[str, torch.Tensor],
    intervention: Mapping[str, torch.Tensor],
    target_ids: torch.Tensor,
    c_token_id: int,
) -> dict[str, object]:
    """Summarize one paired intervention over aligned action rows."""

    if set(correct) != set(ALL_STAGE_NAMES):
        raise ValueError("correct stage set is incomplete")
    if set(intervention) != set(ALL_STAGE_NAMES):
        raise ValueError("intervention stage set is incomplete")
    row_count = correct["raw_logits"].shape[0]
    target_ids = target_ids.detach().to(device="cpu", dtype=torch.long)
    if target_ids.shape != (row_count,):
        raise ValueError("target_ids must align with captured rows")
    if not 0 <= c_token_id < correct["raw_logits"].shape[1]:
        raise ValueError("c_token_id is outside the vocabulary")

    normalized_rms = {
        stage: summarize_values(
            normalized_rms_per_row(correct[stage], intervention[stage])
        )
        for stage in ALL_STAGE_NAMES
    }
    correct_logits = correct["raw_logits"]
    intervention_logits = intervention["raw_logits"]
    rows = torch.arange(row_count)
    correct_argmax = correct_logits.argmax(dim=-1)
    intervention_argmax = intervention_logits.argmax(dim=-1)
    correct_margin = (
        correct_logits[rows, target_ids] - correct_logits[:, c_token_id]
    )
    intervention_margin = (
        intervention_logits[rows, target_ids]
        - intervention_logits[:, c_token_id]
    )
    non_c = target_ids.ne(c_token_id)

    margin: dict[str, object] = {
        "correct": summarize_values(correct_margin),
        "intervention": summarize_values(intervention_margin),
        "intervention_minus_correct": summarize_values(
            intervention_margin - correct_margin
        ),
    }
    if bool(non_c.any()):
        margin["non_c_targets"] = {
            "correct": summarize_values(correct_margin[non_c]),
            "intervention": summarize_values(intervention_margin[non_c]),
            "intervention_minus_correct": summarize_values(
                intervention_margin[non_c] - correct_margin[non_c]
            ),
        }

    return {
        "actions": row_count,
        "normalized_rms": normalized_rms,
        "raw_argmax_change_count": int(
            correct_argmax.ne(intervention_argmax).sum().item()
        ),
        "raw_argmax_change_rate": float(
            correct_argmax.ne(intervention_argmax).float().mean().item()
        ),
        "correct_raw_top1_accuracy": float(
            correct_argmax.eq(target_ids).float().mean().item()
        ),
        "intervention_raw_top1_accuracy": float(
            intervention_argmax.eq(target_ids).float().mean().item()
        ),
        "target_vs_c_logit_margin": margin,
    }


def summarize_conflict_top1(
    target_ids: Sequence[int],
    restricted_top1_ids: Sequence[int],
    global_top1_ids: Sequence[int],
) -> dict[str, float | int]:
    """Summarize restricted and global Top-1 within a prefix conflict group."""

    if not target_ids:
        raise ValueError("cannot summarize an empty conflict group")
    if not (
        len(target_ids) == len(restricted_top1_ids) == len(global_top1_ids)
    ):
        raise ValueError("conflict prediction vectors must have equal lengths")
    restricted_correct = sum(
        int(target) == int(predicted)
        for target, predicted in zip(target_ids, restricted_top1_ids)
    )
    global_correct = sum(
        int(target) == int(predicted)
        for target, predicted in zip(target_ids, global_top1_ids)
    )
    return {
        "rows": len(target_ids),
        "restricted_candidate_top1_correct": restricted_correct,
        "restricted_candidate_top1_accuracy": restricted_correct / len(target_ids),
        "global_raw_top1_correct": global_correct,
        "global_raw_top1_accuracy": global_correct / len(target_ids),
    }


def identical_prefix_conflict_groups(
    actions: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Locate identical revealed prefixes that require different actions."""

    grouped: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for index, action in enumerate(actions):
        position = int(action["position"])
        canvas_ids = tuple(int(value) for value in action["canvas_ids"])
        grouped[(position, canvas_ids[:position])].append(index)

    conflicts = []
    for (position, prefix_ids), indices in sorted(grouped.items()):
        target_ids = sorted({int(actions[index]["target_id"]) for index in indices})
        if len(indices) < 2 or len(target_ids) < 2:
            continue
        conflicts.append(
            {
                "group_id": f"position-{position}-conflict-{len(conflicts)}",
                "position": position,
                "prefix_ids": list(prefix_ids),
                "action_indices": indices,
                "target_ids": target_ids,
            }
        )
    return conflicts
