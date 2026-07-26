#!/usr/bin/env python3
"""Audit SAFE round trips and supported random reveal orders on a SMILES table."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path

import pandas as pd
import torch

from dlm.utils.utils_chem import safe_to_smiles, smiles_to_safe
from marlin.grammar import SafeGrammarMask, _scan
from marlin.tokenizer import load_safe_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--block-width", type=int, default=8)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _record(examples: dict[str, list], key: str, payload: dict) -> None:
    if len(examples[key]) < 10:
        examples[key].append(payload)


def encode_audit_sequence(smiles: str, tokenizer) -> tuple[str, list[int]]:
    """Encode a SMILES exactly as the MARLIN metadata training path does."""

    safe_string = smiles_to_safe(smiles)
    if not safe_string:
        raise ValueError(f"failed to convert SMILES to SAFE: {smiles}")
    token_ids = tokenizer.encode(safe_string, add_special_tokens=True)
    return safe_string, token_ids


def main() -> None:
    args = parse_args()
    git_commit = _git_commit()
    if args.block_width <= 0 or args.trials <= 0:
        raise ValueError("block width and trials must be positive")
    tokenizer = load_safe_tokenizer(args.tokenizer)
    token_strings = [
        tokenizer.convert_ids_to_tokens(index) for index in range(len(tokenizer))
    ]
    special_ids = (
        tokenizer.unk_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.mask_token_id,
    )
    grammar = SafeGrammarMask(
        token_strings,
        lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
        eos_token_id=tokenizer.eos_token_id,
        mask_token_id=tokenizer.mask_token_id,
        special_token_ids=special_ids,
    )
    table = pd.read_csv(args.metadata)
    if args.smiles_column not in table:
        raise KeyError(f"missing SMILES column {args.smiles_column!r}")

    count_keys = (
        "encode_failures",
        "unknown_token_sequences",
        "token_roundtrip_failures",
        "strict_decode_failures",
        "supported_reveal_failures",
        "over_max_length",
    )
    counts: dict[str, int] = {key: 0 for key in count_keys}
    counts.update(
        rows=len(table),
        encoded=0,
        supported_reveal_trials=0,
        max_token_length=0,
    )
    examples: dict[str, list] = {key: [] for key in count_keys}

    for row_index, smiles in enumerate(table[args.smiles_column].astype(str)):
        try:
            safe_string, token_ids = encode_audit_sequence(smiles, tokenizer)
        except Exception as error:
            counts["encode_failures"] += 1
            _record(examples, "encode_failures", {"row": row_index, "error": str(error)})
            continue
        counts["encoded"] += 1
        counts["max_token_length"] = max(counts["max_token_length"], len(token_ids))
        if len(token_ids) > args.max_length:
            counts["over_max_length"] += 1
            _record(
                examples,
                "over_max_length",
                {"row": row_index, "length": len(token_ids)},
            )
        if tokenizer.unk_token_id in token_ids:
            counts["unknown_token_sequences"] += 1
            _record(
                examples,
                "unknown_token_sequences",
                {"row": row_index, "safe": safe_string},
            )
            continue
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        if decoded != safe_string:
            counts["token_roundtrip_failures"] += 1
            _record(
                examples,
                "token_roundtrip_failures",
                {"row": row_index, "safe": safe_string, "decoded": decoded},
            )
            continue
        state = _scan(decoded)
        if safe_to_smiles(decoded, fix=False) is None or state is None or not state.terminal:
            counts["strict_decode_failures"] += 1
            _record(
                examples,
                "strict_decode_failures",
                {"row": row_index, "safe": safe_string},
            )
            continue

        for trial in range(args.trials):
            counts["supported_reveal_trials"] += 1
            rng = random.Random(args.seed + row_index * 17 + trial)
            working = [tokenizer.bos_token_id]
            failure = None
            for block_start in range(1, len(token_ids), args.block_width):
                true_block = token_ids[block_start : block_start + args.block_width]
                block_origin = len(working)
                working.extend([tokenizer.mask_token_id] * len(true_block))
                unresolved = set(range(len(true_block)))
                while unresolved:
                    supported = []
                    for relative_position in unresolved:
                        position = block_origin + relative_position
                        constrained = grammar(
                            working[:position], torch.zeros(len(tokenizer))
                        )
                        if torch.isfinite(
                            constrained[true_block[relative_position]]
                        ):
                            supported.append(relative_position)
                    if not supported:
                        failure = {
                            "row": row_index,
                            "trial": trial,
                            "block_start": block_start,
                            "unresolved": sorted(unresolved),
                            "safe": safe_string,
                        }
                        break
                    chosen = rng.choice(supported)
                    working[block_origin + chosen] = true_block[chosen]
                    unresolved.remove(chosen)
                if failure is not None:
                    break
            if failure is not None or working != token_ids:
                counts["supported_reveal_failures"] += 1
                _record(
                    examples,
                    "supported_reveal_failures",
                    failure
                    or {"row": row_index, "trial": trial, "reason": "token mismatch"},
                )

    manifest = {
        "kind": "MARLIN SAFE supported-reveal oracle",
        "git_commit": git_commit,
        "tokenizer_path": str(args.tokenizer),
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "metadata_path": str(args.metadata),
        "block_width": args.block_width,
        "trials_per_encoded_structure": args.trials,
        "hole_policy": "conservative no-prune except EOS until contiguous prefix",
        "counts": counts,
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
