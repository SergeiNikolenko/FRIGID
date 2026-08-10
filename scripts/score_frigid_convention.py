#!/usr/bin/env python3
"""Score any number of prediction files under FRIGID's benchmark conventions.

Both this reproduction and the FRIGID parity evaluator write the same row schema, so
one scorer over the stored predictions is what makes their numbers comparable. See
`marlin.frigid_convention` for why the Tanimoto denominators differ.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marlin.frigid_convention import score_prediction_file  # noqa: E402

ROWS = (
    ("exact_match_top1", "Exact@1"),
    ("exact_match_top10", "Exact@10"),
    ("tanimoto_top1_mean", "Tanimoto@1"),
    ("tanimoto_top10_mean", "Tanimoto@10"),
    ("tanimoto_mean", "Tanimoto (all cands)"),
    ("formula_match_success_rate", "Formula success"),
    ("candidate_return_rate", "Candidate return"),
    ("never_matched_rate", "Never matched"),
    ("avg_attempts_to_match", "Attempts to match"),
    ("total_spectra", "Spectra"),
    ("truncated_spectra", "Truncated"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "predictions",
        nargs="+",
        help="predictions.jsonl paths, or LABEL=path to name a column",
    )
    parser.add_argument("--json", type=Path, help="also write the raw metrics here")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scored: dict[str, dict] = {}
    for item in args.predictions:
        label, _, path = item.partition("=")
        if not path:
            path, label = label, Path(label).parent.name
        scored[label] = score_prediction_file(path)

    width = max(18, *(len(label) + 2 for label in scored))
    print(f"{'metric':22s}" + "".join(f"{label:>{width}s}" for label in scored))
    for key, title in ROWS:
        line = f"{title:22s}"
        for label in scored:
            value = scored[label].get(key)
            if isinstance(value, bool) or value is None:
                line += f"{'-':>{width}s}"
            elif isinstance(value, int):
                line += f"{value:>{width}d}"
            else:
                line += f"{value:>{width}.4f}"
        print(line)

    print()
    print("Denominators, because a metric name without one is not comparable:")
    for key, title in ROWS:
        for label in scored:
            denominator = scored[label].get("denominators", {}).get(key)
            if denominator:
                print(f"  {title:22s} {denominator}")
                break

    if args.json:
        args.json.write_text(json.dumps(scored, indent=2, sort_keys=True) + "\n")
        print(f"\nwritten: {args.json}")


if __name__ == "__main__":
    main()
