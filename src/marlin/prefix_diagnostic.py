"""Production-aligned diagnostics for sequential MARLIN block actions."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch

from marlin.sampler import MarlinSampler


def build_production_prefix_actions(
    sampler: MarlinSampler,
    target_token_ids: Sequence[int],
    target_mass: float,
) -> list[dict[str, object]]:
    """Build the exact production canvases used to score a target sequence.

    The returned canvases are snapshots taken immediately before revealing the
    target action. This centralizes the fail-closed target validation and the
    production block-extension semantics shared by diagnostics.
    """

    token_ids = [int(token_id) for token_id in target_token_ids]
    if not token_ids or token_ids[0] != sampler.bos_token_id:
        raise ValueError("target sequence must start with BOS")
    if token_ids[-1] != sampler.eos_token_id:
        raise ValueError("target sequence must end with EOS")
    if sampler.eos_token_id in token_ids[1:-1]:
        raise ValueError("target sequence contains EOS before its final token")
    forbidden_targets = set(sampler.forbidden_token_ids) | {sampler.mask_token_id}
    for position, token_id in enumerate(token_ids[1:-1], start=1):
        if token_id in forbidden_targets:
            raise ValueError(
                f"forbidden target token ID {token_id} at position {position}"
            )
    if len(token_ids) > sampler.model.config.max_length:
        raise ValueError("target sequence exceeds model max_length")

    canvas = [sampler.bos_token_id]
    actions: list[dict[str, object]] = []
    for position, target_id in enumerate(token_ids[1:], start=1):
        if position == len(canvas):
            if target_id == sampler.eos_token_id:
                safe = sampler._decode_prefix(canvas)
                smiles = sampler.safe_to_smiles(safe)
                if sampler.constraint.accepts_smiles(smiles, target_mass):
                    break
            block_width = sampler._next_block_width(len(canvas))
            if block_width <= 0:
                raise ValueError("production canvas ended before target EOS")
            canvas.extend([sampler.mask_token_id] * block_width)
        if canvas[position] != sampler.mask_token_id:
            raise AssertionError("current production action is not unresolved")

        actions.append(
            {
                "position": position,
                "block_index": (
                    (position - 1) // sampler.model.config.block_width
                ),
                "block_offset": (
                    (position - 1) % sampler.model.config.block_width
                ),
                "canvas_length": len(canvas),
                "canvas_ids": tuple(canvas),
                "target_id": target_id,
            }
        )
        canvas[position] = target_id
        if target_id == sampler.eos_token_id:
            break
    return actions


def _distribution_record(
    logits: torch.Tensor,
    target_id: int,
    token_string: Callable[[int], str],
) -> dict[str, float | int | str | bool | None]:
    finite = torch.isfinite(logits)
    target_allowed = bool(finite[target_id].item())
    if not bool(finite.any().item()):
        return {
            "target_allowed": target_allowed,
            "target_probability": 0.0,
            "target_rank": None,
            "top1_id": None,
            "top1_token": None,
            "top1_probability": None,
        }

    usable = logits.masked_fill(~finite, -torch.inf)
    probabilities = usable.softmax(dim=-1)
    top1_id = int(probabilities.argmax().item())
    if target_allowed:
        target_probability = float(probabilities[target_id].item())
        target_rank = int((usable > usable[target_id]).sum().item()) + 1
    else:
        target_probability = 0.0
        target_rank = None
    return {
        "target_allowed": target_allowed,
        "target_probability": target_probability,
        "target_rank": target_rank,
        "top1_id": top1_id,
        "top1_token": token_string(top1_id),
        "top1_probability": float(probabilities[top1_id].item()),
    }


def summarize_prefix_actions(actions: Sequence[dict]) -> dict[str, object]:
    """Return compact reconstruction statistics for action-level records."""

    total = len(actions)
    if total == 0:
        raise ValueError("cannot summarize an empty action sequence")
    target_allowed = sum(bool(action["target_allowed"]) for action in actions)
    raw_top1 = sum(
        action["raw"]["top1_id"] == action["target_id"] for action in actions
    )
    raw_top10 = sum(
        action["raw"]["target_rank"] is not None
        and action["raw"]["target_rank"] <= 10
        for action in actions
    )
    constrained_top1 = sum(
        action["constrained"]["top1_id"] == action["target_id"]
        for action in actions
    )
    constrained_top10 = sum(
        action["constrained"]["target_rank"] is not None
        and action["constrained"]["target_rank"] <= 10
        for action in actions
    )
    first_disallowed_action = next(
        (action for action in actions if not action["target_allowed"]),
        None,
    )
    first_disallowed = (
        None
        if first_disallowed_action is None
        else {
            **(
                {"metadata_row": int(first_disallowed_action["metadata_row"])}
                if "metadata_row" in first_disallowed_action
                else {}
            ),
            "position": int(first_disallowed_action["position"]),
        }
    )
    return {
        "actions": total,
        "target_allowed": target_allowed,
        "target_allowed_rate": target_allowed / total,
        "constraint_dead_end_actions": sum(
            action["constrained"]["top1_id"] is None for action in actions
        ),
        "first_disallowed_position": (
            None if first_disallowed is None else first_disallowed["position"]
        ),
        "first_disallowed": first_disallowed,
        "raw_top1_correct": raw_top1,
        "raw_top1_accuracy": raw_top1 / total,
        "raw_top10_correct": raw_top10,
        "raw_top10_accuracy": raw_top10 / total,
        "constrained_top1_correct": constrained_top1,
        "constrained_top1_accuracy": constrained_top1 / total,
        "constrained_top10_correct": constrained_top10,
        "constrained_top10_accuracy": constrained_top10 / total,
        "mean_raw_target_probability": sum(
            float(action["raw"]["target_probability"]) for action in actions
        )
        / total,
        "mean_constrained_target_probability": sum(
            float(action["constrained"]["target_probability"]) for action in actions
        )
        / total,
    }


def summarize_prefix_rows(rows: Sequence[dict]) -> dict[str, object]:
    """Aggregate row actions while retaining the first failure's row identity."""

    actions = [
        {**action, "metadata_row": int(row["metadata_row"])}
        for row in rows
        for action in row["actions"]
    ]
    return summarize_prefix_actions(actions)


@torch.no_grad()
def diagnose_production_prefix(
    sampler: MarlinSampler,
    target_token_ids: Sequence[int],
    fingerprint: torch.Tensor,
    target_mass: float,
    token_string: Callable[[int], str],
    *,
    temperature: float = 1.0,
) -> dict[str, object]:
    """Score true left-to-right actions on the exact production block canvas.

    Each block is appended at its full production width. The current leftmost
    mask is scored, the true token is revealed, and the model is run again.
    This continues across block boundaries through the target EOS action.
    """

    if temperature <= 0:
        raise ValueError("temperature must be positive")

    device = next(sampler.model.parameters()).device
    conditioned = fingerprint.to(device=device, dtype=torch.float32).reshape(1, -1)
    mass = torch.tensor([target_mass], device=device, dtype=torch.float32)
    actions: list[dict[str, object]] = []

    for action in build_production_prefix_actions(
        sampler,
        target_token_ids,
        target_mass,
    ):
        position = int(action["position"])
        target_id = int(action["target_id"])
        canvas = list(action["canvas_ids"])
        input_ids = torch.tensor([canvas], device=device, dtype=torch.long)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = sampler._sampling_logits(input_ids, mass, conditioned)
        raw_logits = logits[0, position].float() / temperature
        prefix_ids = canvas[:position]
        constrained_logits = sampler.constrain_action_logits(
            prefix_ids,
            raw_logits,
            target_mass,
        )
        raw = _distribution_record(raw_logits, target_id, token_string)
        constrained = _distribution_record(
            constrained_logits,
            target_id,
            token_string,
        )
        actions.append(
            {
                "position": action["position"],
                "block_index": action["block_index"],
                "block_offset": action["block_offset"],
                "canvas_length": action["canvas_length"],
                "target_id": target_id,
                "target_token": token_string(target_id),
                "target_allowed": bool(constrained["target_allowed"]),
                "raw": raw,
                "constrained": constrained,
            }
        )
    return {
        "summary": summarize_prefix_actions(actions),
        "actions": actions,
    }
