#!/usr/bin/env python3
"""Build deterministic, balanced and locked NPLIB1 validation panels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from marlin.benchmark_selection import hash_spec_names


NAMESPACE = "marlin-nplib1-validation-v1"
CONTINUOUS = ("neutral_mass", "active_bits", "heavy_atoms", "replicate_count")
BALANCE_COLUMNS = (
    "neutral_mass_bin",
    "active_bits_bin",
    "heavy_atoms_bin",
    "replicate_count_bin",
    "has_phosphorus",
    "has_sulfur",
    "has_halogen",
)
ACCEPTANCE = {
    "micro32": {"max_smd": 0.15, "max_category_share_difference": 0.05},
    "micro64": {"max_smd": 0.12, "max_category_share_difference": 0.05},
    "micro128": {"max_smd": 0.10, "max_category_share_difference": 0.03},
    "macro64": {"max_smd": 0.15, "max_category_share_difference": 0.05},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    return hashlib.sha256(
        "\0".join(str(part) for part in parts).encode()
    ).hexdigest()


def quantile_bins(series: pd.Series, bins: int = 4) -> pd.Series:
    ranked = series.rank(method="first", pct=True)
    return np.minimum((ranked * bins).astype(int), bins - 1).astype(str)


def load_population(
    metadata_path: Path,
    fingerprints_path: Path,
    fingerprint_key: str,
    threshold: float,
) -> pd.DataFrame:
    frame = pd.read_csv(metadata_path)
    required = {"spec_name", "inchikey_first_block", "neutral_mass", "smiles"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Metadata lacks columns: {sorted(missing)}")
    frame = frame.copy()
    frame["spec_name"] = frame["spec_name"].astype(str)
    frame["inchikey_first_block"] = (
        frame["inchikey_first_block"].astype(str).str.split("-", n=1).str[0]
    )
    if frame["spec_name"].duplicated().any():
        raise ValueError("Metadata contains duplicate spec_name values")
    if (frame["inchikey_first_block"] == "").any():
        raise ValueError("Metadata contains empty molecule connectivity")

    with np.load(fingerprints_path, allow_pickle=False) as bundle:
        if fingerprint_key not in bundle:
            raise KeyError(f"{fingerprint_key!r} absent from {fingerprints_path}")
        values = np.asarray(bundle[fingerprint_key])
        if "spectrum_ids" in bundle:
            positions = {
                str(name): index for index, name in enumerate(bundle["spectrum_ids"])
            }
            missing_names = [name for name in frame["spec_name"] if name not in positions]
            if missing_names:
                raise ValueError(
                    f"Fingerprint archive lacks spectra: {missing_names[:5]}"
                )
            values = values[[positions[name] for name in frame["spec_name"]]]
    if values.shape != (len(frame), 4096):
        raise ValueError(
            f"Fingerprints must have shape ({len(frame)}, 4096), got {values.shape}"
        )
    frame["active_bits"] = (values >= threshold).sum(axis=1).astype(float)

    molecule_features = []
    for spec_name, smiles in zip(frame["spec_name"], frame["smiles"], strict=True):
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            raise ValueError(f"Invalid SMILES for {spec_name}")
        symbols = {atom.GetSymbol() for atom in molecule.GetAtoms()}
        molecule_features.append(
            {
                "heavy_atoms": molecule.GetNumHeavyAtoms(),
                "has_phosphorus": int("P" in symbols),
                "has_sulfur": int("S" in symbols),
                "has_halogen": int(bool(symbols.intersection({"F", "Cl", "Br", "I"}))),
            }
        )
    frame = pd.concat(
        [frame, pd.DataFrame(molecule_features, index=frame.index)], axis=1
    )
    counts = frame["inchikey_first_block"].value_counts()
    frame["replicate_count"] = frame["inchikey_first_block"].map(counts).astype(float)
    for column in ("neutral_mass", "active_bits", "heavy_atoms", "replicate_count"):
        frame[f"{column}_bin"] = quantile_bins(frame[column])
    return frame


def balance_matrix(
    target: pd.DataFrame, candidates: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blocks, proportions, weights = [], [], []
    for column in BALANCE_COLUMNS:
        categories = sorted(
            set(target[column].astype(str)).union(candidates[column].astype(str))
        )
        target_props = np.asarray(
            [(target[column].astype(str) == value).mean() for value in categories]
        )
        blocks.append(
            np.column_stack(
                [
                    (candidates[column].astype(str) == value).to_numpy(float)
                    for value in categories
                ]
            )
        )
        proportions.extend(target_props)
        variance = np.maximum(target_props * (1.0 - target_props), 0.02)
        weights.extend((1.0 / len(categories)) / variance)
    return (
        np.concatenate(blocks, axis=1),
        np.asarray(proportions),
        np.asarray(weights),
    )


def select_balanced(
    target: pd.DataFrame,
    candidates: pd.DataFrame,
    size: int,
    namespace: str,
) -> pd.DataFrame:
    if not 0 < size <= len(candidates):
        raise ValueError(f"Cannot select {size} rows from {len(candidates)} candidates")
    candidates = candidates.sort_values("spec_name", kind="stable").reset_index(drop=True)
    matrix, target_props, weights = balance_matrix(target, candidates)
    selected = np.zeros(len(candidates), dtype=bool)
    counts = np.zeros(matrix.shape[1])
    row_norm = (matrix * matrix * weights).sum(axis=1)
    jitter = np.asarray(
        [
            int(stable_hash(namespace, name)[:15], 16) / float(16**15)
            for name in candidates["spec_name"]
        ]
    )
    indices: list[int] = []
    for step in range(1, size + 1):
        residual = counts - step * target_props
        scores = row_norm + matrix @ (2.0 * weights * residual) + jitter * 1e-9
        scores[selected] = np.inf
        chosen = int(np.argmin(scores))
        selected[chosen] = True
        indices.append(chosen)
        counts += matrix[chosen]
    return candidates.iloc[indices].reset_index(drop=True)


def representatives(frame: pd.DataFrame, namespace: str) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["_hash"] = [
        stable_hash(namespace, block, spec)
        for block, spec in zip(
            ranked["inchikey_first_block"], ranked["spec_name"], strict=True
        )
    ]
    return (
        ranked.sort_values(["inchikey_first_block", "_hash"], kind="stable")
        .drop_duplicates("inchikey_first_block")
        .drop(columns="_hash")
        .reset_index(drop=True)
    )


def discrepancy(target: pd.DataFrame, panel: pd.DataFrame) -> dict[str, Any]:
    smd = {}
    for column in CONTINUOUS:
        denominator = float(target[column].std(ddof=0))
        difference = float(panel[column].mean() - target[column].mean())
        smd[column] = abs(difference / denominator) if denominator else 0.0
    category_differences = {}
    for column in BALANCE_COLUMNS:
        target_shares = target[column].astype(str).value_counts(normalize=True)
        panel_shares = panel[column].astype(str).value_counts(normalize=True)
        categories = set(target_shares.index).union(panel_shares.index)
        category_differences[column] = max(
            abs(float(target_shares.get(value, 0) - panel_shares.get(value, 0)))
            for value in categories
        )
    return {
        "absolute_standardized_mean_difference": smd,
        "max_absolute_smd": max(smd.values()),
        "max_category_share_difference": max(category_differences.values()),
    }


def write_panel(frame: pd.DataFrame, path: Path, panel: str) -> dict[str, Any]:
    output = pd.DataFrame(
        {
            "spec_name": frame["spec_name"],
            "inchikey_first_block": frame["inchikey_first_block"],
            "neutral_mass": frame["neutral_mass"],
            "selection_hash": [
                stable_hash(NAMESPACE, panel, name) for name in frame["spec_name"]
            ],
            "panel": panel,
            "selection_order": np.arange(1, len(frame) + 1),
        }
    )
    output.to_csv(path, sep="\t", index=False)
    return {
        "path": path.name,
        "rows": len(output),
        "sha256": sha256_file(path),
        "ordered_spec_names_sha256": hash_spec_names(output["spec_name"].tolist()),
    }


def build_benchmarks(
    val_metadata: Path,
    val_fingerprints: Path,
    test_metadata: Path,
    test_fingerprints: Path,
    output_dir: Path,
    fingerprint_key: str = "probs",
    threshold: float = 0.95,
    micro_sizes: tuple[int, ...] = (32, 64, 128),
    macro_size: int = 64,
) -> dict[str, Any]:
    if tuple(sorted(set(micro_sizes))) != micro_sizes:
        raise ValueError("micro_sizes must be strictly increasing")
    val = load_population(
        val_metadata, val_fingerprints, fingerprint_key, threshold
    )
    test = load_population(
        test_metadata, test_fingerprints, fingerprint_key, threshold
    )
    micro = select_balanced(val, val, micro_sizes[-1], f"{NAMESPACE}:micro")
    micro_blocks = set(micro["inchikey_first_block"])
    unique_val = representatives(val, f"{NAMESPACE}:macro-target")
    macro_candidates = representatives(
        val[~val["inchikey_first_block"].isin(micro_blocks)],
        f"{NAMESPACE}:macro-candidates",
    )
    macro = select_balanced(
        unique_val, macro_candidates, macro_size, f"{NAMESPACE}:macro"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs: dict[str, Any] = {}
    for size in micro_sizes:
        name = f"nplib1_val_micro{size}_v1"
        outputs[name] = write_panel(micro.iloc[:size], output_dir / f"{name}.tsv", name)
    macro_name = f"nplib1_val_macro{macro_size}_v1"
    outputs[macro_name] = write_panel(
        macro, output_dir / f"{macro_name}.tsv", macro_name
    )
    full_val_name = f"nplib1_val_full{len(val)}_v1"
    outputs[full_val_name] = write_panel(
        val, output_dir / f"{full_val_name}.tsv", full_val_name
    )
    full_test_name = f"nplib1_test_locked_full{len(test)}_v1"
    outputs[full_test_name] = write_panel(
        test, output_dir / f"{full_test_name}.tsv", full_test_name
    )

    checks = {
        **{
            f"micro{size}_vs_val": discrepancy(val, micro.iloc[:size])
            for size in micro_sizes
        },
        f"macro{macro_size}_vs_unique_val": discrepancy(unique_val, macro),
    }
    acceptance = {}
    for panel in [*(f"micro{size}" for size in micro_sizes), f"macro{macro_size}"]:
        limits = ACCEPTANCE.get(panel)
        check = checks[
            f"{panel}_vs_unique_val" if panel.startswith("macro") else f"{panel}_vs_val"
        ]
        accepted = limits is None or (
            check["max_absolute_smd"] <= limits["max_smd"]
            and check["max_category_share_difference"]
            <= limits["max_category_share_difference"]
        )
        acceptance[panel] = {
            **(limits or {}),
            "threshold_configured": limits is not None,
            "accepted": accepted,
        }
    if not all(result["accepted"] for result in acceptance.values()):
        raise ValueError(f"Panel distribution acceptance failed: {acceptance}")
    report = {
        "schema_version": 1,
        "selection_namespace": NAMESPACE,
        "inputs": {
            "val_metadata": {"path": str(val_metadata), "sha256": sha256_file(val_metadata)},
            "val_fingerprints": {
                "path": str(val_fingerprints),
                "sha256": sha256_file(val_fingerprints),
                "key": fingerprint_key,
                "threshold": threshold,
            },
            "test_metadata": {
                "path": str(test_metadata),
                "sha256": sha256_file(test_metadata),
            },
            "test_fingerprints": {
                "path": str(test_fingerprints),
                "sha256": sha256_file(test_fingerprints),
                "key": fingerprint_key,
                "threshold": threshold,
            },
        },
        "counts": {
            "validation_rows": len(val),
            "validation_unique_molecules": int(val["inchikey_first_block"].nunique()),
            "test_rows": len(test),
            "test_unique_molecules": int(test["inchikey_first_block"].nunique()),
        },
        "quality_checks": {
            "micro_panels_nested": True,
            "micro_macro_spec_overlap": len(
                set(micro["spec_name"]).intersection(macro["spec_name"])
            ),
            "micro_macro_molecule_overlap": len(
                micro_blocks.intersection(macro["inchikey_first_block"])
            ),
            "target_fields_used_for_model_scoring": [],
            "labels_used_only_for_panel_balancing": [
                "SMILES-derived heavy atom and element-presence features"
            ],
            "test_panel_locked_until_final_promotion": True,
            "distribution_acceptance": acceptance,
            "all_panels_accepted": True,
        },
        "distribution_checks": checks,
        "outputs": outputs,
    }
    (output_dir / "nplib1_selection_report_v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-metadata", type=Path, required=True)
    parser.add_argument("--val-fingerprints", type=Path, required=True)
    parser.add_argument("--test-metadata", type=Path, required=True)
    parser.add_argument("--test-fingerprints", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fingerprint-key", default="probs")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--micro-sizes", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--macro-size", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_benchmarks(
        args.val_metadata.resolve(),
        args.val_fingerprints.resolve(),
        args.test_metadata.resolve(),
        args.test_fingerprints.resolve(),
        args.output_dir.resolve(),
        args.fingerprint_key,
        args.threshold,
        tuple(args.micro_sizes),
        args.macro_size,
    )
    print(json.dumps(report["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
