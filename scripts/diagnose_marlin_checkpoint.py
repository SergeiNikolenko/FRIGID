#!/usr/bin/env python3
"""Measure teacher-forced block reconstruction for a MARLIN checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

from dlm.utils.utils_chem import smiles_to_safe
from evaluate_marlin_nplib1 import load_decoder
from marlin.tokenizer import load_safe_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--rows", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--layer0-long-residual-scale", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_decoder(
        args.checkpoint,
        device,
        use_ema=not args.no_ema,
        layer0_long_residual_scale=args.layer0_long_residual_scale,
    )
    tokenizer = load_safe_tokenizer(args.tokenizer)
    metadata = pd.read_csv(args.metadata).iloc[args.row : args.row + args.rows]
    if metadata.empty:
        raise ValueError("requested metadata range is empty")
    prepared = []
    for metadata_index, record in metadata.iterrows():
        molecule = Chem.MolFromSmiles(str(record["smiles"]))
        if molecule is None:
            raise ValueError(f"invalid target SMILES at row {metadata_index}")
        safe = smiles_to_safe(str(record["smiles"]))
        encoded = tokenizer(safe, return_tensors="pt")["input_ids"][0]
        fingerprint = AllChem.GetMorganGenerator(radius=2, fpSize=4096).GetFingerprint(
            molecule
        )
        fingerprint_array = np.zeros(4096, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_array)
        prepared.append(
            {
                "metadata_index": int(metadata_index),
                "record": record,
                "safe": safe,
                "encoded": encoded,
                "fingerprint": torch.from_numpy(fingerprint_array).to(device).unsqueeze(0),
                "mass": torch.tensor([Descriptors.ExactMolWt(molecule)], device=device),
            }
        )

    rows = []
    aggregate_variants = {
        name: {
            "correct_at_1": 0,
            "correct_at_10": 0,
            "total": 0,
            "target_probability_sum": 0.0,
            "target_nll_sum": 0.0,
            "kl_from_correct_sum": 0.0,
        }
        for name in ("correct", "shuffled", "zero")
    }
    for prepared_index, item in enumerate(prepared):
        metadata_index = item["metadata_index"]
        record = item["record"]
        safe = item["safe"]
        encoded = item["encoded"]
        fingerprint_tensor = item["fingerprint"]
        mass = item["mass"]
        shuffled_fingerprint = prepared[
            (prepared_index + 1) % len(prepared)
        ]["fingerprint"]
        if len(prepared) == 1:
            shuffled_fingerprint = torch.roll(fingerprint_tensor, shifts=1, dims=1)
        fingerprint_variants = {
            "correct": fingerprint_tensor,
            "shuffled": shuffled_fingerprint,
            "zero": torch.zeros_like(fingerprint_tensor),
        }
        blocks = []
        row_variants = {
            name: {
                "correct_at_1": 0,
                "correct_at_10": 0,
                "total": 0,
                "target_probability_sum": 0.0,
                "target_nll_sum": 0.0,
                "kl_from_correct_sum": 0.0,
            }
            for name in fingerprint_variants
        }
        start = 1
        while start < len(encoded):
            boundary = (
                ((start - 1) // model.config.block_width) + 1
            ) * model.config.block_width + 1
            end = min(boundary, len(encoded))
            input_ids = (
                torch.cat(
                    (
                        encoded[:start],
                        torch.full(
                            (end - start,), tokenizer.mask_token_id, dtype=torch.long
                        ),
                    )
                )
                .to(device)
                .unsqueeze(0)
            )
            targets = encoded[start:end].to(device)
            variant_logits = {}
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ),
            ):
                for name, variant_fingerprint in fingerprint_variants.items():
                    variant_logits[name] = model.sampling_logits(
                        input_ids, mass, variant_fingerprint
                    )[0, start:end].float()
            correct_log_probabilities = F.log_softmax(
                variant_logits["correct"], dim=-1
            )
            correct_probabilities = correct_log_probabilities.exp()
            for name, logits in variant_logits.items():
                log_probabilities = F.log_softmax(logits, dim=-1)
                probabilities = log_probabilities.exp()
                target_probabilities = probabilities.gather(
                    1, targets.unsqueeze(1)
                ).squeeze(1)
                target_ranks = (
                    (logits > logits.gather(1, targets.unsqueeze(1))).sum(dim=1)
                    + 1
                )
                metrics = row_variants[name]
                metrics["correct_at_1"] += int(target_ranks.eq(1).sum())
                metrics["correct_at_10"] += int(target_ranks.le(10).sum())
                metrics["total"] += len(targets)
                metrics["target_probability_sum"] += float(
                    target_probabilities.sum()
                )
                metrics["target_nll_sum"] += float(
                    -log_probabilities.gather(1, targets.unsqueeze(1)).sum()
                )
                metrics["kl_from_correct_sum"] += float(
                    (
                        correct_probabilities
                        * (correct_log_probabilities - log_probabilities)
                    )
                    .sum(dim=1)
                    .sum()
                )

            logits = variant_logits["correct"]
            probabilities = correct_probabilities
            predictions = logits.argmax(dim=-1)
            positions = []
            for offset, target in enumerate(targets.tolist()):
                target_probability = float(probabilities[offset, target])
                target_rank = int((logits[offset] > logits[offset, target]).sum()) + 1
                predicted = int(predictions[offset])
                positions.append(
                    {
                        "position": start + offset,
                        "target_id": target,
                        "target_token": tokenizer.convert_ids_to_tokens(target),
                        "target_probability": target_probability,
                        "target_rank": target_rank,
                        "predicted_id": predicted,
                        "predicted_token": tokenizer.convert_ids_to_tokens(predicted),
                    }
                )
            blocks.append({"start": start, "end": end, "positions": positions})
            start = end
        summarized_variants = {}
        for name, metrics in row_variants.items():
            total = metrics["total"]
            summarized_variants[name] = {
                "top1_accuracy": metrics["correct_at_1"] / total,
                "top10_accuracy": metrics["correct_at_10"] / total,
                "mean_target_probability": metrics["target_probability_sum"] / total,
                "mean_target_nll": metrics["target_nll_sum"] / total,
                "mean_kl_from_correct": metrics["kl_from_correct_sum"] / total,
            }
            for key, value in metrics.items():
                aggregate_variants[name][key] += value
        correct_metrics = row_variants["correct"]
        total = correct_metrics["total"]
        rows.append(
            {
                "metadata_row": metadata_index,
                "spec_name": str(record["spec_name"]),
                "safe": safe,
                "token_count": len(encoded),
                "teacher_forced_top1_correct": correct_metrics["correct_at_1"],
                "teacher_forced_top10_correct": correct_metrics["correct_at_10"],
                "teacher_forced_total": total,
                "teacher_forced_top1_accuracy": correct_metrics["correct_at_1"]
                / total,
                "teacher_forced_top10_accuracy": correct_metrics["correct_at_10"]
                / total,
                "mean_target_probability": correct_metrics[
                    "target_probability_sum"
                ]
                / total,
                "conditioning_variants": summarized_variants,
                "blocks": blocks,
            }
        )

    total = sum(row["teacher_forced_total"] for row in rows)
    conditioning_variants = {}
    for name, metrics in aggregate_variants.items():
        variant_total = metrics["total"]
        conditioning_variants[name] = {
            "top1_accuracy": metrics["correct_at_1"] / variant_total,
            "top10_accuracy": metrics["correct_at_10"] / variant_total,
            "mean_target_probability": metrics["target_probability_sum"]
            / variant_total,
            "mean_target_nll": metrics["target_nll_sum"] / variant_total,
            "mean_kl_from_correct": metrics["kl_from_correct_sum"]
            / variant_total,
        }
    result = {
        "checkpoint": str(args.checkpoint),
        "ema": not args.no_ema,
        "block_width": model.config.block_width,
        "row_start": args.row,
        "row_count": len(rows),
        "teacher_forced_top1_correct": sum(
            row["teacher_forced_top1_correct"] for row in rows
        ),
        "teacher_forced_top10_correct": sum(
            row["teacher_forced_top10_correct"] for row in rows
        ),
        "teacher_forced_total": total,
        "teacher_forced_top1_accuracy": sum(
            row["teacher_forced_top1_correct"] for row in rows
        )
        / total,
        "teacher_forced_top10_accuracy": sum(
            row["teacher_forced_top10_correct"] for row in rows
        )
        / total,
        "mean_target_probability": sum(
            row["mean_target_probability"] * row["teacher_forced_total"] for row in rows
        )
        / total,
        "conditioning_variants": conditioning_variants,
        "conditioning_signal": {
            "correct_vs_shuffled_target_nll_gain": (
                conditioning_variants["shuffled"]["mean_target_nll"]
                - conditioning_variants["correct"]["mean_target_nll"]
            ),
            "correct_vs_zero_target_nll_gain": (
                conditioning_variants["zero"]["mean_target_nll"]
                - conditioning_variants["correct"]["mean_target_nll"]
            ),
            "correct_vs_shuffled_mean_kl": conditioning_variants["shuffled"][
                "mean_kl_from_correct"
            ],
            "correct_vs_zero_mean_kl": conditioning_variants["zero"][
                "mean_kl_from_correct"
            ],
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "rows"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
