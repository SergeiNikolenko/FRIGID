#!/usr/bin/env python3
"""Score MARLIN checkpoints on true production-style sequential actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

from audit_marlin_safe_oracle import encode_audit_sequence
from dlm.utils.utils_chem import safe_to_smiles
from evaluate_marlin_nplib1 import load_decoder
from marlin.grammar import SafeGrammarMask
from marlin.mass_shell import MassShellConstraint
from marlin.prefix_diagnostic import (
    diagnose_production_prefix,
    summarize_prefix_actions,
)
from marlin.sampler import MarlinSampler
from marlin.token_properties import build_token_property_table
from marlin.tokenizer import load_safe_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--expected-block-width", type=int)
    return parser.parse_args()


def build_production_sampler(model, tokenizer) -> MarlinSampler:
    special_ids = {
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.mask_token_id,
        tokenizer.pad_token_id,
    }
    token_masses, token_atoms, token_valences = build_token_property_table(
        len(tokenizer), tokenizer.convert_ids_to_tokens, special_ids
    )
    constraint = MassShellConstraint(
        token_masses,
        token_atoms,
        token_valences,
        ppm_tolerance=10.0,
        valence_slack=4.0,
        eos_boost=1.0,
        eos_token_id=tokenizer.eos_token_id,
    )
    grammar = SafeGrammarMask(
        [tokenizer.convert_ids_to_tokens(index) for index in range(len(tokenizer))],
        lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
        eos_token_id=tokenizer.eos_token_id,
        mask_token_id=tokenizer.mask_token_id,
        special_token_ids=tuple(special_ids) + (tokenizer.unk_token_id,),
        ppm_tolerance=10.0,
        valence_slack=4.0,
    )
    return MarlinSampler(
        model,
        constraint,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        mask_token_id=tokenizer.mask_token_id,
        decode_tokens=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
        safe_to_smiles=lambda safe: safe_to_smiles(safe, fix=True),
        strict_safe_to_smiles=lambda safe: safe_to_smiles(safe, fix=False),
        grammar_mask=grammar,
        forbidden_token_ids=(
            tokenizer.unk_token_id,
            tokenizer.bos_token_id,
            tokenizer.mask_token_id,
            tokenizer.pad_token_id,
        ),
        mass_shell_enabled=True,
        generation_mode="block",
    )


def oracle_condition(safe: str, fingerprint_bits: int) -> tuple[torch.Tensor, float]:
    smiles = safe_to_smiles(safe, fix=False)
    molecule = Chem.MolFromSmiles(smiles) if smiles else None
    if molecule is None:
        raise ValueError("training SAFE target does not strictly decode")
    fingerprint = AllChem.GetMorganGenerator(
        radius=2,
        fpSize=fingerprint_bits,
    ).GetFingerprint(molecule)
    array = np.zeros(fingerprint_bits, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, array)
    return torch.from_numpy(array), float(Descriptors.ExactMolWt(molecule))


def main() -> None:
    args = parse_args()
    if args.row < 0 or args.rows <= 0:
        raise ValueError("row must be non-negative and rows must be positive")
    device = torch.device(args.device)
    model = load_decoder(
        args.checkpoint,
        device,
        use_ema=args.weights == "ema",
    )
    if (
        args.expected_block_width is not None
        and model.config.block_width != args.expected_block_width
    ):
        raise ValueError(
            f"checkpoint block width is {model.config.block_width}; "
            f"expected {args.expected_block_width}"
        )
    tokenizer = load_safe_tokenizer(args.tokenizer)
    sampler = build_production_sampler(model, tokenizer)
    metadata = pd.read_csv(args.metadata).iloc[args.row : args.row + args.rows]
    if metadata.empty:
        raise ValueError("requested metadata range is empty")
    if "smiles" not in metadata:
        raise KeyError("metadata must contain a smiles column")

    rows = []
    all_actions = []
    for metadata_index, record in metadata.iterrows():
        safe, target_token_ids = encode_audit_sequence(
            str(record["smiles"]),
            tokenizer,
        )
        fingerprint, target_mass = oracle_condition(
            safe,
            model.config.fingerprint_bits,
        )
        diagnostic = diagnose_production_prefix(
            sampler,
            target_token_ids,
            fingerprint,
            target_mass,
            lambda token_id: str(tokenizer.convert_ids_to_tokens(token_id)),
            temperature=args.temperature,
        )
        actions = diagnostic["actions"]
        all_actions.extend(actions)
        rows.append(
            {
                "metadata_row": int(metadata_index),
                "spec_name": (
                    str(record["spec_name"]) if "spec_name" in record else None
                ),
                "smiles": str(record["smiles"]),
                "safe": safe,
                "token_count_including_bos_eos": len(target_token_ids),
                "target_mass": target_mass,
                "summary": diagnostic["summary"],
                "actions": actions,
            }
        )

    result = {
        "kind": "MARLIN production-style sequential prefix diagnostic",
        "checkpoint": str(args.checkpoint),
        "weights": args.weights,
        "tokenizer": str(args.tokenizer),
        "metadata": str(args.metadata),
        "row_start": args.row,
        "row_count": len(rows),
        "block_width": model.config.block_width,
        "temperature": args.temperature,
        "conditioning": "oracle Morgan radius=2 fingerprint and exact molecular mass",
        "constraints": {
            "grammar": "SafeGrammarMask",
            "mass_shell": True,
            "ppm_tolerance": 10.0,
            "valence_slack": 4.0,
            "eos_boost": 1.0,
        },
        "aggregate": summarize_prefix_actions(all_actions),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "rows"},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
