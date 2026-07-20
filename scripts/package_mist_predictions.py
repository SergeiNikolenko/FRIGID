#!/usr/bin/env python3
"""Package official MIST Morgan-4096 probabilities in benchmark order."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_predictions(
    predictions: Path,
    metadata: Path,
    formula_manifest: Path,
    output: Path,
) -> dict:
    with predictions.open("rb") as handle:
        payload = pickle.load(handle)
    names = [str(value) for value in payload["names"]]
    probabilities = np.asarray(payload["preds"], dtype=np.float32)
    if probabilities.shape != (len(names), 4096):
        raise ValueError(f"Expected Morgan-4096 probabilities, got {probabilities.shape}")
    if len(names) != len(set(names)):
        raise ValueError("MIST predictions contain duplicate spectrum IDs")
    ordered = pd.read_csv(metadata).sort_values("fingerprint_index", kind="stable")
    expected = ordered["spec_name"].astype(str).tolist()
    positions = {name: index for index, name in enumerate(names)}
    missing = [name for name in expected if name not in positions]
    extra = sorted(set(names) - set(expected))
    if missing or extra:
        raise ValueError(
            f"MIST/metadata ID mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )
    values = probabilities[[positions[name] for name in expected]]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, probs=values, spectrum_ids=np.asarray(expected))
    formula_evidence = json.loads(formula_manifest.read_text())
    if (
        formula_evidence.get("kind")
        != "MIST-CF top-1 predicted-formula bridge into official MIST"
        or formula_evidence.get("formula_source")
        != "MIST-CF top-1 prediction; no ground-truth formula"
        or formula_evidence.get("rows") != len(expected)
    ):
        raise ValueError("Formula manifest does not prove the formula-blind MIST-CF lane")
    manifest = {
        "schema_version": 1,
        "kind": "official MIST fingerprint probabilities from MIST-CF predicted formulas",
        "rows": len(expected),
        "fingerprint_bits": 4096,
        "formula_source": formula_evidence["formula_source"],
        "predictions_pickle_sha256": sha256_file(predictions),
        "reference_metadata_sha256": sha256_file(metadata),
        "formula_manifest_sha256": sha256_file(formula_manifest),
        "output_sha256": sha256_file(output),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--formula-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(package_predictions(args.predictions, args.metadata, args.formula_manifest, args.output)))


if __name__ == "__main__":
    main()
