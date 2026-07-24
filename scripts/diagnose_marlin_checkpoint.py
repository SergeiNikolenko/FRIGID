#!/usr/bin/env python3
"""Measure teacher-forced block reconstruction for a MARLIN checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_decoder(args.checkpoint, device, use_ema=not args.no_ema)
    tokenizer = load_safe_tokenizer(args.tokenizer)
    metadata = pd.read_csv(args.metadata).iloc[args.row : args.row + args.rows]
    if metadata.empty:
        raise ValueError("requested metadata range is empty")
    rows = []
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
        fingerprint_tensor = torch.from_numpy(fingerprint_array).to(device).unsqueeze(0)
        mass = torch.tensor([Descriptors.ExactMolWt(molecule)], device=device)

        blocks = []
        correct_at_1 = 0
        correct_at_10 = 0
        total = 0
        target_probability_sum = 0.0
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
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ),
            ):
                logits = model.sampling_logits(input_ids, mass, fingerprint_tensor)[
                    0, start:end
                ].float()
            targets = encoded[start:end].to(device)
            predictions = logits.argmax(dim=-1)
            probabilities = logits.softmax(dim=-1)
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
                correct_at_1 += int(target_rank == 1)
                correct_at_10 += int(target_rank <= 10)
                target_probability_sum += target_probability
            blocks.append({"start": start, "end": end, "positions": positions})
            total += end - start
            start = end
        rows.append(
            {
                "metadata_row": int(metadata_index),
                "spec_name": str(record["spec_name"]),
                "safe": safe,
                "token_count": len(encoded),
                "teacher_forced_top1_correct": correct_at_1,
                "teacher_forced_top10_correct": correct_at_10,
                "teacher_forced_total": total,
                "teacher_forced_top1_accuracy": correct_at_1 / total,
                "teacher_forced_top10_accuracy": correct_at_10 / total,
                "mean_target_probability": target_probability_sum / total,
                "blocks": blocks,
            }
        )

    total = sum(row["teacher_forced_total"] for row in rows)
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
