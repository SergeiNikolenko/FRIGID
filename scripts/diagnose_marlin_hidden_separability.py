#!/usr/bin/env python3
"""Locate where MARLIN loses production-prefix or fingerprint conditioning."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import pandas as pd
import torch

from audit_marlin_safe_oracle import encode_audit_sequence
from diagnose_marlin_production_prefix import (
    build_production_sampler,
    oracle_condition,
)
from evaluate_marlin_nplib1 import git_commit, load_decoder, sha256
from marlin.hidden_separability import (
    ALL_STAGE_NAMES,
    capture_action_stages,
    identical_prefix_conflict_groups,
    normalized_rms_per_row,
    summarize_conflict_top1,
    summarize_intervention,
    summarize_values,
    symmetric_normalized_rms,
)
from marlin.prefix_diagnostic import build_production_prefix_actions
from marlin.tokenizer import load_safe_tokenizer, validate_safe_tokenizer


INTERVENTIONS = ("correct", "context_ablation", "fingerprint_swap")
FOCUS_POSITIONS = (8, 14)
SAME_MASS_ABSOLUTE_TOLERANCE_DA = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        dest="checkpoints",
        required=True,
        help="checkpoint path; repeat to compare checkpoints on identical actions",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="checkpoint label; either omit all labels or provide one per checkpoint",
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--expected-metadata-sha256", required=True)
    parser.add_argument("--expected-action-count", type=int, required=True)
    parser.add_argument(
        "--expected-conflict-position",
        type=int,
        action="append",
        required=True,
        help="expected identical-prefix conflict position; repeat as needed",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("ema", "raw"), default="raw")
    parser.add_argument("--expected-block-width", type=int, required=True)
    return parser.parse_args()


def checkpoint_labels(args: argparse.Namespace) -> list[str]:
    if args.label and len(args.label) != len(args.checkpoints):
        raise ValueError("provide exactly one --label per --checkpoint, or none")
    labels = args.label or [
        f"checkpoint_{index}_{path.parent.name or path.stem}"
        for index, path in enumerate(args.checkpoints)
    ]
    if len(set(labels)) != len(labels):
        raise ValueError("checkpoint labels must be unique")
    return labels


def token_id(tokenizer, token: str) -> int:
    value = tokenizer.convert_tokens_to_ids(token)
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"token {token!r} did not map to one ID")
        value = value[0]
    value = int(value)
    if value == tokenizer.unk_token_id:
        raise ValueError(f"required token {token!r} maps to UNK")
    if str(tokenizer.convert_ids_to_tokens(value)) != token:
        raise ValueError(f"required token {token!r} does not round-trip")
    return value


def token_string(tokenizer, value: int) -> str:
    return str(tokenizer.convert_ids_to_tokens(int(value)))


def model_contract(model) -> dict[str, object]:
    return asdict(model.config)


def validate_model_and_tokenizer(
    model,
    tokenizer,
    *,
    expected_block_width: int,
) -> None:
    if model.config.block_width != expected_block_width:
        raise ValueError(
            f"checkpoint block width is {model.config.block_width}; "
            f"expected {expected_block_width}"
        )
    validate_safe_tokenizer(
        tokenizer,
        expected_vocab_size=model.config.vocab_size,
        expected_special_token_ids={
            "unk": tokenizer.unk_token_id,
            "bos": model.config.bos_token_id,
            "eos": model.config.eos_token_id,
            "pad": model.config.pad_token_id,
            "mask": model.config.mask_token_id,
        },
    )


def selected_metadata(path: Path, row: int, rows: int) -> pd.DataFrame:
    if row < 0 or rows < 2:
        raise ValueError("row must be non-negative and rows must be at least two")
    metadata = pd.read_csv(path)
    if "smiles" not in metadata:
        raise KeyError("metadata must contain a smiles column")
    selected = metadata.iloc[row : row + rows]
    if len(selected) != rows:
        raise ValueError(
            f"requested exactly {rows} metadata rows at offset {row}; "
            f"found {len(selected)}"
        )
    return selected


def build_conditions_and_actions(model, tokenizer, metadata, sampler):
    conditions = []
    actions: list[dict[str, object]] = []
    for row_offset, (metadata_index, record) in enumerate(metadata.iterrows()):
        safe, target_token_ids = encode_audit_sequence(
            str(record["smiles"]),
            tokenizer,
        )
        fingerprint, target_mass = oracle_condition(
            safe,
            model.config.fingerprint_bits,
        )
        conditions.append(
            {
                "metadata_row": int(metadata_index),
                "row_offset": row_offset,
                "spec_name": (
                    str(record["spec_name"]) if "spec_name" in record else None
                ),
                "smiles": str(record["smiles"]),
                "safe": safe,
                "target_mass": target_mass,
                "fingerprint": fingerprint,
            }
        )
        row_actions = build_production_prefix_actions(
            sampler,
            target_token_ids,
            target_mass,
        )
        for action in row_actions:
            canvas_ids = tuple(int(value) for value in action["canvas_ids"])
            position = int(action["position"])
            target_id_value = int(action["target_id"])
            unconstrained = torch.zeros(
                model.config.vocab_size,
                dtype=torch.float32,
                device=next(model.parameters()).device,
            )
            constrained = sampler.constrain_action_logits(
                list(canvas_ids[:position]),
                unconstrained,
                target_mass,
            )
            if not bool(torch.isfinite(constrained[target_id_value]).item()):
                raise ValueError(
                    "production constraint rejects target at "
                    f"metadata row {metadata_index}, position {position}, "
                    f"token ID {target_id_value}"
                )
            actions.append(
                {
                    **action,
                    "canvas_ids": canvas_ids,
                    "metadata_row": int(metadata_index),
                    "row_offset": row_offset,
                    "target_allowed": True,
                }
            )

    masses = [float(condition["target_mass"]) for condition in conditions]
    mass_span = max(masses) - min(masses)
    if mass_span > SAME_MASS_ABSOLUTE_TOLERANCE_DA:
        raise ValueError(
            "cyclic fingerprint intervention requires same-mass rows; "
            f"mass span is {mass_span:.12g} Da"
        )
    for index, condition in enumerate(conditions):
        swapped = conditions[(index + 1) % len(conditions)]
        if torch.equal(condition["fingerprint"], swapped["fingerprint"]):
            raise ValueError(
                "cyclic fingerprint intervention is identical for metadata rows "
                f"{condition['metadata_row']} and {swapped['metadata_row']}"
            )
        condition["fingerprint_swap_row"] = int(swapped["metadata_row"])
        condition["fingerprint_swap"] = swapped["fingerprint"]
    if not actions:
        raise ValueError("selected metadata produced no production actions")
    return conditions, actions, mass_span


def run_checkpoint(model, actions, conditions, *, device, batch_size):
    outputs = {
        intervention: {
            stage: [None] * len(actions) for stage in ALL_STAGE_NAMES
        }
        for intervention in INTERVENTIONS
    }
    by_canvas_length: dict[int, list[int]] = defaultdict(list)
    for index, action in enumerate(actions):
        by_canvas_length[int(action["canvas_length"])].append(index)

    for canvas_length, indices in sorted(by_canvas_length.items()):
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            input_ids = torch.tensor(
                [actions[index]["canvas_ids"] for index in batch_indices],
                dtype=torch.long,
                device=device,
            )
            if input_ids.shape[1] != canvas_length:
                raise AssertionError("canvas-length grouping is inconsistent")
            positions = torch.tensor(
                [int(actions[index]["position"]) for index in batch_indices],
                dtype=torch.long,
                device=device,
            )
            masses = torch.tensor(
                [
                    float(conditions[int(actions[index]["row_offset"])]["target_mass"])
                    for index in batch_indices
                ],
                dtype=torch.float32,
                device=device,
            )
            fingerprints = torch.stack(
                [
                    conditions[int(actions[index]["row_offset"])]["fingerprint"]
                    for index in batch_indices
                ]
            ).to(device=device, dtype=torch.float32)
            swapped_fingerprints = torch.stack(
                [
                    conditions[int(actions[index]["row_offset"])]["fingerprint_swap"]
                    for index in batch_indices
                ]
            ).to(device=device, dtype=torch.float32)
            ablated_ids = torch.full_like(input_ids, model.config.mask_token_id)
            ablated_ids[:, 0] = model.config.bos_token_id

            captured = {
                "correct": capture_action_stages(
                    model,
                    input_ids,
                    masses,
                    fingerprints,
                    positions,
                ),
                "context_ablation": capture_action_stages(
                    model,
                    ablated_ids,
                    masses,
                    fingerprints,
                    positions,
                ),
                "fingerprint_swap": capture_action_stages(
                    model,
                    input_ids,
                    masses,
                    swapped_fingerprints,
                    positions,
                ),
            }
            for local_index, action_index in enumerate(batch_indices):
                for intervention in INTERVENTIONS:
                    for stage in ALL_STAGE_NAMES:
                        outputs[intervention][stage][action_index] = captured[
                            intervention
                        ][stage][local_index]

    return {
        intervention: {
            stage: torch.stack(values)
            for stage, values in stage_outputs.items()
        }
        for intervention, stage_outputs in outputs.items()
    }


def prediction_record(logits, target_id_value, c_token_id, tokenizer):
    probabilities = logits.softmax(dim=-1)
    top1_id = int(logits.argmax().item())
    target_rank = int((logits > logits[target_id_value]).sum().item()) + 1
    return {
        "raw_top1_id": top1_id,
        "raw_top1_token": token_string(tokenizer, top1_id),
        "raw_top1_probability": float(probabilities[top1_id].item()),
        "target_probability": float(probabilities[target_id_value].item()),
        "target_rank": target_rank,
        "target_logit": float(logits[target_id_value].item()),
        "c_logit": float(logits[c_token_id].item()),
        "target_vs_c_logit_margin": float(
            (logits[target_id_value] - logits[c_token_id]).item()
        ),
    }


def action_records(captures, actions, conditions, c_token_id, tokenizer):
    records = []
    for index, action in enumerate(actions):
        target_id_value = int(action["target_id"])
        condition = conditions[int(action["row_offset"])]
        predictions = {
            intervention: prediction_record(
                captures[intervention]["raw_logits"][index],
                target_id_value,
                c_token_id,
                tokenizer,
            )
            for intervention in INTERVENTIONS
        }
        deltas = {}
        for intervention in INTERVENTIONS[1:]:
            deltas[intervention] = {
                "normalized_rms": {
                    stage: float(
                        normalized_rms_per_row(
                            captures["correct"][stage][index].reshape(1, -1),
                            captures[intervention][stage][index].reshape(1, -1),
                        )[0].item()
                    )
                    for stage in ALL_STAGE_NAMES
                },
                "raw_argmax_changed": (
                    predictions["correct"]["raw_top1_id"]
                    != predictions[intervention]["raw_top1_id"]
                ),
                "target_vs_c_margin_delta": (
                    predictions[intervention]["target_vs_c_logit_margin"]
                    - predictions["correct"]["target_vs_c_logit_margin"]
                ),
            }
        records.append(
            {
                "action_index": index,
                "metadata_row": int(action["metadata_row"]),
                "fingerprint_swap_metadata_row": int(
                    condition["fingerprint_swap_row"]
                ),
                "position": int(action["position"]),
                "block_index": int(action["block_index"]),
                "block_offset": int(action["block_offset"]),
                "canvas_length": int(action["canvas_length"]),
                "prefix_ids": list(
                    action["canvas_ids"][: int(action["position"])]
                ),
                "target_id": target_id_value,
                "target_token": token_string(tokenizer, target_id_value),
                "target_allowed": bool(action["target_allowed"]),
                "predictions": predictions,
                "paired_deltas": deltas,
            }
        )
    return records


def correct_summary(captures, actions, c_token_id, tokenizer):
    logits = captures["correct"]["raw_logits"]
    target_ids = torch.tensor([int(action["target_id"]) for action in actions])
    argmax_ids = logits.argmax(dim=-1)
    rows = torch.arange(len(actions))
    target_probabilities = logits.softmax(dim=-1)[rows, target_ids]
    target_ranks = torch.stack(
        [
            (logits[index] > logits[index, target_id_value]).sum() + 1
            for index, target_id_value in enumerate(target_ids)
        ]
    ).float()
    histogram = Counter(int(value) for value in argmax_ids.tolist())
    return {
        "actions": len(actions),
        "raw_top1_correct": int(argmax_ids.eq(target_ids).sum().item()),
        "raw_top1_accuracy": float(argmax_ids.eq(target_ids).float().mean().item()),
        "raw_top1_is_c_count": int(argmax_ids.eq(c_token_id).sum().item()),
        "raw_top1_is_c_rate": float(argmax_ids.eq(c_token_id).float().mean().item()),
        "mean_target_probability": float(target_probabilities.mean().item()),
        "target_rank": summarize_values(target_ranks),
        "raw_top1_histogram": [
            {
                "token_id": value,
                "token": token_string(tokenizer, value),
                "count": count,
            }
            for value, count in sorted(
                histogram.items(), key=lambda item: (-item[1], item[0])
            )
        ],
    }


def conflict_analyses(conflicts, captures, actions, c_token_id, tokenizer):
    analyses = []
    for conflict in conflicts:
        indices = [int(value) for value in conflict["action_indices"]]
        candidate_ids = sorted({c_token_id, *conflict["target_ids"]})
        rows = []
        for index in indices:
            action = actions[index]
            target_id_value = int(action["target_id"])
            variants = {}
            for intervention in INTERVENTIONS:
                logits = captures[intervention]["raw_logits"][index]
                restricted_id = max(candidate_ids, key=lambda value: float(logits[value]))
                variants[intervention] = {
                    **prediction_record(
                        logits,
                        target_id_value,
                        c_token_id,
                        tokenizer,
                    ),
                    "restricted_candidate_top1_id": restricted_id,
                    "restricted_candidate_top1_token": token_string(
                        tokenizer, restricted_id
                    ),
                }
            rows.append(
                {
                    "action_index": index,
                    "metadata_row": int(action["metadata_row"]),
                    "target_id": target_id_value,
                    "target_token": token_string(tokenizer, target_id_value),
                    "variants": variants,
                }
            )

        intervention_summary = {}
        for intervention in INTERVENTIONS:
            intervention_summary[intervention] = summarize_conflict_top1(
                [row["target_id"] for row in rows],
                [
                    row["variants"][intervention][
                        "restricted_candidate_top1_id"
                    ]
                    for row in rows
                ],
                [
                    row["variants"][intervention]["raw_top1_id"]
                    for row in rows
                ],
            )

        pairwise = []
        for first_index, second_index in combinations(indices, 2):
            first_target = int(actions[first_index]["target_id"])
            second_target = int(actions[second_index]["target_id"])
            if first_target == second_target:
                continue
            first_logits = captures["correct"]["raw_logits"][first_index]
            second_logits = captures["correct"]["raw_logits"][second_index]
            pairwise.append(
                {
                    "metadata_rows": [
                        int(actions[first_index]["metadata_row"]),
                        int(actions[second_index]["metadata_row"]),
                    ],
                    "target_ids": [first_target, second_target],
                    "target_tokens": [
                        token_string(tokenizer, first_target),
                        token_string(tokenizer, second_target),
                    ],
                    "correct_conditioning_symmetric_normalized_rms": {
                        stage: symmetric_normalized_rms(
                            captures["correct"][stage][first_index],
                            captures["correct"][stage][second_index],
                        )
                        for stage in ALL_STAGE_NAMES
                    },
                    "raw_argmax_differs": bool(
                        first_logits.argmax().item() != second_logits.argmax().item()
                    ),
                    "first_own_vs_other_target_margin": float(
                        (first_logits[first_target] - first_logits[second_target]).item()
                    ),
                    "second_own_vs_other_target_margin": float(
                        (second_logits[second_target] - second_logits[first_target]).item()
                    ),
                }
            )
        analyses.append(
            {
                "group_id": conflict["group_id"],
                "position": int(conflict["position"]),
                "candidate_ids_including_c": candidate_ids,
                "candidate_tokens_including_c": [
                    token_string(tokenizer, value) for value in candidate_ids
                ],
                "intervention_summary": intervention_summary,
                "rows": rows,
                "different_target_pairs": pairwise,
            }
        )
    return analyses


def focus_position_analyses(captures, actions, c_token_id, tokenizer):
    result = {}
    for position in FOCUS_POSITIONS:
        indices = [
            index
            for index, action in enumerate(actions)
            if int(action["position"]) == position
        ]
        result[str(position)] = {
            "present": bool(indices),
            "actions": [
                {
                    "action_index": index,
                    "metadata_row": int(actions[index]["metadata_row"]),
                    "target_id": int(actions[index]["target_id"]),
                    "target_token": token_string(
                        tokenizer, int(actions[index]["target_id"])
                    ),
                    "predictions": {
                        intervention: prediction_record(
                            captures[intervention]["raw_logits"][index],
                            int(actions[index]["target_id"]),
                            c_token_id,
                            tokenizer,
                        )
                        for intervention in INTERVENTIONS
                    },
                }
                for index in indices
            ],
        }
    return result


def serializable_conflicts(conflicts, actions, tokenizer):
    return [
        {
            "group_id": conflict["group_id"],
            "position": int(conflict["position"]),
            "prefix_ids": conflict["prefix_ids"],
            "prefix_tokens": [
                token_string(tokenizer, value) for value in conflict["prefix_ids"]
            ],
            "action_indices": conflict["action_indices"],
            "metadata_rows": [
                int(actions[index]["metadata_row"])
                for index in conflict["action_indices"]
            ],
            "target_ids": conflict["target_ids"],
            "target_tokens": [
                token_string(tokenizer, value) for value in conflict["target_ids"]
            ],
        }
        for conflict in conflicts
    ]


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.expected_block_width <= 0:
        raise ValueError("expected-block-width must be positive")
    if args.expected_action_count <= 0:
        raise ValueError("expected-action-count must be positive")
    expected_conflict_positions = sorted(args.expected_conflict_position)
    if any(position <= 0 for position in expected_conflict_positions):
        raise ValueError("expected-conflict-position must be positive")
    if len(set(expected_conflict_positions)) != len(expected_conflict_positions):
        raise ValueError("expected-conflict-position values must be unique")
    expected_metadata_sha256 = args.expected_metadata_sha256.lower()
    if len(expected_metadata_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in expected_metadata_sha256
    ):
        raise ValueError("expected-metadata-sha256 must be 64 hexadecimal digits")
    metadata_sha256 = sha256(args.metadata)
    if metadata_sha256 != expected_metadata_sha256:
        raise ValueError(
            f"metadata SHA256 is {metadata_sha256}; "
            f"expected {expected_metadata_sha256}"
        )
    labels = checkpoint_labels(args)
    metadata = selected_metadata(args.metadata, args.row, args.rows)
    tokenizer = load_safe_tokenizer(args.tokenizer)
    c_token_id = token_id(tokenizer, "c")
    device = torch.device(args.device)

    shared_contract = None
    conditions = None
    actions = None
    conflicts = None
    mass_span = None
    checkpoint_results = []
    for label, checkpoint_path in zip(labels, args.checkpoints):
        model = load_decoder(
            checkpoint_path,
            device,
            use_ema=args.weights == "ema",
        ).eval().float()
        validate_model_and_tokenizer(
            model,
            tokenizer,
            expected_block_width=args.expected_block_width,
        )
        contract = model_contract(model)
        if shared_contract is None:
            shared_contract = contract
            sampler = build_production_sampler(model, tokenizer)
            conditions, actions, mass_span = build_conditions_and_actions(
                model,
                tokenizer,
                metadata,
                sampler,
            )
            conflicts = identical_prefix_conflict_groups(actions)
            if len(actions) != args.expected_action_count:
                raise ValueError(
                    f"selected rows produced {len(actions)} actions; "
                    f"expected {args.expected_action_count}"
                )
            actual_conflict_positions = sorted(
                {int(conflict["position"]) for conflict in conflicts}
            )
            if actual_conflict_positions != expected_conflict_positions:
                raise ValueError(
                    "identical-prefix conflict positions are "
                    f"{actual_conflict_positions}; expected "
                    f"{expected_conflict_positions}"
                )
            del sampler
        elif contract != shared_contract:
            raise ValueError(
                f"checkpoint {checkpoint_path} has a different decoder config"
            )

        captures = run_checkpoint(
            model,
            actions,
            conditions,
            device=device,
            batch_size=args.batch_size,
        )
        target_ids = torch.tensor(
            [int(action["target_id"]) for action in actions],
            dtype=torch.long,
        )
        checkpoint_results.append(
            {
                "label": label,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": sha256(checkpoint_path),
                "weights": args.weights,
                "correct_conditioning": correct_summary(
                    captures,
                    actions,
                    c_token_id,
                    tokenizer,
                ),
                "paired_interventions": {
                    intervention: summarize_intervention(
                        captures["correct"],
                        captures[intervention],
                        target_ids,
                        c_token_id,
                    )
                    for intervention in INTERVENTIONS[1:]
                },
                "identical_prefix_conflict_groups": conflict_analyses(
                    conflicts,
                    captures,
                    actions,
                    c_token_id,
                    tokenizer,
                ),
                "focus_positions": focus_position_analyses(
                    captures,
                    actions,
                    c_token_id,
                    tokenizer,
                ),
                "actions": action_records(
                    captures,
                    actions,
                    conditions,
                    c_token_id,
                    tokenizer,
                ),
            }
        )
        del captures, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "kind": "MARLIN hidden separability paired-intervention diagnostic",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "tokenizer": str(args.tokenizer),
        "tokenizer_sha256": sha256(args.tokenizer),
        "metadata": str(args.metadata),
        "metadata_sha256": metadata_sha256,
        "expected_metadata_sha256": expected_metadata_sha256,
        "row_start": args.row,
        "row_count": args.rows,
        "action_count": len(actions),
        "expected_action_count": args.expected_action_count,
        "expected_conflict_positions": expected_conflict_positions,
        "all_targets_allowed": all(
            bool(action["target_allowed"]) for action in actions
        ),
        "expected_block_width": args.expected_block_width,
        "decoder_config": shared_contract,
        "device": str(device),
        "inference_dtype": "float32",
        "autocast": False,
        "batch_size": args.batch_size,
        "normalized_rms_definition": (
            "RMS(correct-intervention)/max(RMS(correct),1e-12); "
            "reported per action then summarized"
        ),
        "interventions": {
            "context_ablation": (
                "replace every non-BOS token on the current production canvas "
                "with MASK; retain correct mass and fingerprint"
            ),
            "fingerprint_swap": (
                "retain the correct production canvas and mass; cyclically swap "
                "Morgan radius=2 fingerprints between selected same-mass rows"
            ),
        },
        "conditioning": (
            "oracle Morgan radius=2 fingerprint and exact molecular mass"
        ),
        "same_mass_absolute_tolerance_da": SAME_MASS_ABSOLUTE_TOLERANCE_DA,
        "selected_mass_span_da": mass_span,
        "c_token_id": c_token_id,
        "c_token": "c",
        "metadata_rows": [
            {
                "metadata_row": int(condition["metadata_row"]),
                "spec_name": condition["spec_name"],
                "smiles": condition["smiles"],
                "safe": condition["safe"],
                "target_mass": float(condition["target_mass"]),
                "fingerprint_active_bits": int(
                    condition["fingerprint"].sum().item()
                ),
                "fingerprint_swap_metadata_row": int(
                    condition["fingerprint_swap_row"]
                ),
            }
            for condition in conditions
        ],
        "identical_prefix_conflict_groups": serializable_conflicts(
            conflicts,
            actions,
            tokenizer,
        ),
        "checkpoints": checkpoint_results,
    }
    if not result["all_targets_allowed"]:
        raise AssertionError("hidden separability cannot report disallowed targets")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "kind": result["kind"],
                "output": str(args.output),
                "row_count": result["row_count"],
                "action_count": result["action_count"],
                "all_targets_allowed": result["all_targets_allowed"],
                "identical_prefix_conflict_groups": len(conflicts),
                "checkpoints": [
                    {
                        "label": checkpoint["label"],
                        "correct_conditioning": checkpoint[
                            "correct_conditioning"
                        ],
                        "paired_interventions": checkpoint[
                            "paired_interventions"
                        ],
                    }
                    for checkpoint in checkpoint_results
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
