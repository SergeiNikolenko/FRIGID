#!/usr/bin/env python3
"""Audit SAFE token lengths in the pinned MARLIN training stream."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
from rdkit import Chem

from dlm.utils.utils_chem import safe_to_smiles
from marlin.tokenizer import load_safe_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-records", type=int, default=1_000_000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--exclude-inchikeys", type=Path, required=True)
    return parser.parse_args()


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    if args.max_records <= 0 or args.max_length <= 0:
        raise ValueError("max records and max length must be positive")

    tokenizer = load_safe_tokenizer(args.tokenizer)
    exclusions_table = pd.read_csv(args.exclude_inchikeys)
    exclusion_column = (
        "inchi" if "inchi" in exclusions_table else "inchikey"
    )
    excluded_keys = {
        str(value).split("-")[0]
        for value in exclusions_table[exclusion_column].dropna()
    }
    stream = datasets.load_dataset(
        args.dataset,
        revision=args.revision,
        split=args.split,
        streaming=True,
        cache_dir=args.cache_dir,
    )
    lengths: list[int] = []
    missing_safe = 0
    strict_decode_failures = 0
    excluded_test_structures = 0
    eligible_records = 0
    overlength_examples: list[dict[str, int | str]] = []
    strict_decode_examples: list[dict[str, int | str]] = []
    excluded_examples: list[dict[str, int | str]] = []
    for row_index, example in enumerate(stream):
        if row_index >= args.max_records:
            break
        safe_string = example.get("safe", example.get("input"))
        if not safe_string:
            missing_safe += 1
            continue
        length = len(tokenizer.encode(safe_string, add_special_tokens=True))
        lengths.append(length)
        if length > args.max_length and len(overlength_examples) < 20:
            overlength_examples.append(
                {
                    "row": row_index,
                    "length": length,
                    "safe_prefix": str(safe_string)[:160],
                }
            )
        if length > args.max_length:
            continue
        smiles = safe_to_smiles(safe_string, fix=False)
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
        if molecule is None:
            strict_decode_failures += 1
            if len(strict_decode_examples) < 20:
                strict_decode_examples.append(
                    {"row": row_index, "safe_prefix": str(safe_string)[:160]}
                )
            continue
        key = Chem.MolToInchiKey(molecule).split("-")[0]
        if key in excluded_keys:
            excluded_test_structures += 1
            if len(excluded_examples) < 20:
                excluded_examples.append({"row": row_index, "inchikey": key})
            continue
        eligible_records += 1

    if not lengths:
        raise RuntimeError("training stream audit did not find any SAFE sequences")
    values = np.asarray(lengths, dtype=np.int32)
    histogram = Counter(int(value) for value in values)
    manifest = {
        "schema_version": 2,
        "kind": "MARLIN pinned training-stream eligibility audit",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "dataset": args.dataset,
        "dataset_revision": args.revision,
        "split": args.split,
        "requested_records": args.max_records,
        "examined_records": int(len(values) + missing_safe),
        "tokenized_records": int(len(values)),
        "missing_safe_records": missing_safe,
        "strict_safe_decode": True,
        "strict_decode_failures_within_length": strict_decode_failures,
        "excluded_test_structures_within_length": excluded_test_structures,
        "eligible_records": eligible_records,
        "maximum_allowed_length": args.max_length,
        "overlength_records": int(np.count_nonzero(values > args.max_length)),
        "maximum_observed_length": int(values.max()),
        "length_quantiles": {
            str(quantile): float(np.quantile(values, quantile))
            for quantile in (0.5, 0.9, 0.99, 0.999, 1.0)
        },
        "length_histogram": dict(sorted(histogram.items())),
        "overlength_examples": overlength_examples,
        "strict_decode_examples": strict_decode_examples,
        "excluded_examples": excluded_examples,
        "tokenizer_path": str(args.tokenizer),
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "exclusion_path": str(args.exclude_inchikeys),
        "exclusion_sha256": hashlib.sha256(
            args.exclude_inchikeys.read_bytes()
        ).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
