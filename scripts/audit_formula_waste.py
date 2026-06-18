#!/usr/bin/env python3
"""Summarize formula waste from FRIGID detailed benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute formula waste diagnostics from FRIGID detailed_results.csv"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to detailed_results.csv",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to write a JSON summary",
    )
    parser.add_argument(
        "--output-md",
        help="Optional path to write a short markdown summary",
    )
    return parser.parse_args()


def _float_or_zero(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    with input_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        summary = {"input": str(input_path), "total_spectra": 0}
    else:
        total_valid = [_float_or_zero(r, "total_valid") for r in rows]
        total_generated = [_float_or_zero(r, "total_generated") for r in rows]
        total_formula_matched = [_float_or_zero(r, "total_formula_matched") for r in rows]
        unique_valid = [_float_or_zero(r, "unique_valid_smiles") for r in rows]
        duplicate_valid = [_float_or_zero(r, "duplicate_valid_smiles") for r in rows]
        formula_match_fraction_among_valid = [
            (matched / valid) if valid else 0.0
            for matched, valid in zip(total_formula_matched, total_valid)
        ]
        unique_valid_fraction_among_valid = [
            (uniq / valid) if valid else 0.0
            for uniq, valid in zip(unique_valid, total_valid)
        ]
        valid_duplicate_rate = [
            (dup / valid) if valid else 0.0
            for dup, valid in zip(duplicate_valid, total_valid)
        ]
        summary = {
            "input": str(input_path),
            "total_spectra": len(rows),
            "avg_total_generated": mean(total_generated),
            "avg_total_valid": mean(total_valid),
            "avg_formula_matches": mean(total_formula_matched),
            "avg_unique_valid_smiles": mean(unique_valid),
            "avg_duplicate_valid_smiles": mean(duplicate_valid),
            "avg_valid_duplicate_rate": mean(valid_duplicate_rate),
            "avg_formula_match_fraction_among_valid": mean(formula_match_fraction_among_valid),
            "avg_unique_valid_fraction_among_valid": mean(unique_valid_fraction_among_valid),
        }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(
            "\n".join(
                [
                    "# FRIGID Formula Waste Audit",
                    "",
                    f"- Input: `{input_path}`",
                    f"- Total spectra: `{summary.get('total_spectra', 0)}`",
                    f"- Avg valid / case: `{summary.get('avg_total_valid', 0):.2f}`",
                    f"- Avg formula matches / case: `{summary.get('avg_formula_matches', 0):.2f}`",
                    f"- Avg formula match fraction among valid: `{summary.get('avg_formula_match_fraction_among_valid', 0)*100:.2f}%`",
                    f"- Avg unique valid fraction among valid: `{summary.get('avg_unique_valid_fraction_among_valid', 0)*100:.2f}%`",
                    f"- Avg valid duplicate rate: `{summary.get('avg_valid_duplicate_rate', 0)*100:.2f}%`",
                    "",
                ]
            )
        )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
