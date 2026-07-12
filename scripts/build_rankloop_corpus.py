#!/usr/bin/env python
"""Build a deterministic train-only RankLoop candidate corpus."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from frigid.rankloop_corpus import (  # noqa: E402
    build_rankloop_corpus,
    load_candidate_source,
    load_spectrum_records,
    write_rankloop_corpus,
)


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Source must use NAME=PATH syntax.")
    name, raw_path = value.split("=", maxsplit=1)
    name = name.strip()
    path = Path(raw_path).expanduser().resolve()
    if not name:
        raise argparse.ArgumentTypeError("Source name cannot be empty.")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Source file does not exist: {path}")
    return name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe RankLoop training and development candidate lists.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--labels-tsv", required=True)
    parser.add_argument("--split-tsv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--source", action="append", type=parse_source, default=[])
    parser.add_argument("--development-fraction", type=float, default=0.1)
    parser.add_argument("--negatives-per-query", type=int, default=32)
    parser.add_argument("--max-spectra-per-molecule", type=int, default=4)
    parser.add_argument("--max-query-spectra", type=int)
    parser.add_argument("--fingerprint-bits", type=int, default=2048)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels_path = Path(args.labels_tsv).expanduser().resolve()
    split_path = Path(args.split_tsv).expanduser().resolve()
    records, record_rejections = load_spectrum_records(
        labels_path,
        split_path,
        dataset_split=args.dataset_split,
        fingerprint_bits=args.fingerprint_bits,
        fingerprint_radius=args.fingerprint_radius,
    )
    allowed_queries = {record.spec_name for record in records}
    combined_sources = defaultdict(list)
    source_stats: dict[str, dict[str, int]] = {}
    for source_name, source_path in args.source:
        source_rows, stats = load_candidate_source(
            source_name,
            source_path,
            allowed_queries=allowed_queries,
            fingerprint_bits=args.fingerprint_bits,
            fingerprint_radius=args.fingerprint_radius,
        )
        for query, rows in source_rows.items():
            combined_sources[query].extend(rows)
        source_stats[source_name] = stats

    frame, report = build_rankloop_corpus(
        records,
        source_candidates=dict(combined_sources),
        development_fraction=args.development_fraction,
        negatives_per_query=args.negatives_per_query,
        max_spectra_per_molecule=args.max_spectra_per_molecule,
        max_query_spectra=args.max_query_spectra,
        seed=args.seed,
    )
    report["record_rejections"] = record_rejections
    report["source_statistics"] = source_stats
    parameters = {
        key: value
        for key, value in vars(args).items()
        if key != "source"
    }
    parameters["source_names"] = [name for name, _ in args.source]
    manifest = write_rankloop_corpus(
        frame,
        report,
        output_dir=args.output_dir,
        labels_tsv=labels_path,
        split_tsv=split_path,
        source_paths=args.source,
        parameters=parameters,
        repo_root=PROJECT_ROOT,
    )
    print(json.dumps(manifest["outputs"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
