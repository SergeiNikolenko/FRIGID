#!/usr/bin/env python
"""Build locked row-representative and molecule-balanced MSG benchmark panels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
SELECTION_NAMESPACE = "frigid-msg-compact-v1"
CONTINUOUS_FEATURES = (
    "parentmass",
    "peak_count",
    "spectral_entropy",
    "replicate_count",
    "mist_active_bits",
    "formula_heavy_atoms",
)
BALANCE_COLUMNS = (
    "instrument_group",
    "ionization_group",
    "parentmass_bin",
    "peak_count_bin",
    "spectral_entropy_bin",
    "replicate_count_bin",
    "mist_active_bits_bin",
    "has_phosphorus",
    "has_sulfur",
    "has_halogen",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(*parts: object) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_inchi_block(value: object) -> str:
    return str(value).strip().split("-", maxsplit=1)[0]


def load_exclusion_manifests(paths: Iterable[Path]) -> set[str]:
    names: set[str] = set()
    for path in paths:
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
        frame = pd.read_csv(path, sep=delimiter)
        if "spec_name" not in frame:
            raise ValueError(f"Exclusion manifest lacks spec_name: {path}")
        current = frame["spec_name"].astype(str)
        if current.duplicated().any():
            raise ValueError(f"Exclusion manifest has duplicate spec_name: {path}")
        overlap = names.intersection(current)
        if overlap:
            raise ValueError(f"Exclusion manifests overlap at {sorted(overlap)[:5]}")
        names.update(current)
    return names


FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_features(formula: str) -> dict[str, int]:
    normalized = re.sub(r"[+-]$", "", str(formula).strip())
    counts: dict[str, int] = {}
    position = 0
    for match in FORMULA_TOKEN.finditer(normalized):
        if match.start() != position:
            raise ValueError(f"Unsupported molecular formula: {formula!r}")
        element, raw_count = match.groups()
        counts[element] = counts.get(element, 0) + int(raw_count or 1)
        position = match.end()
    if position != len(normalized) or not counts:
        raise ValueError(f"Unsupported molecular formula: {formula!r}")
    return {
        "formula_heavy_atoms": sum(
            count for element, count in counts.items() if element != "H"
        ),
        "has_phosphorus": int(counts.get("P", 0) > 0),
        "has_sulfur": int(counts.get("S", 0) > 0),
        "has_halogen": int(
            any(counts.get(element, 0) > 0 for element in ("F", "Cl", "Br", "I"))
        ),
    }


def read_spectrum_features(path: Path) -> tuple[float, int, float]:
    parentmass: float | None = None
    intensities: list[float] = []
    in_peaks = False
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith(">parentmass "):
                parentmass = float(line.split(maxsplit=1)[1])
            elif line == ">ms2peaks":
                in_peaks = True
            elif in_peaks and line and not line.startswith((">", "#")):
                fields = line.split()
                if len(fields) >= 2:
                    intensities.append(float(fields[1]))
    if parentmass is None:
        raise ValueError(f"Spectrum lacks parentmass: {path}")
    values = np.asarray(intensities, dtype=np.float64)
    if values.size and float(values.sum()) > 0:
        probabilities = values / values.sum()
        entropy = float(
            -(probabilities * np.log(probabilities + np.finfo(float).eps)).sum()
        )
    else:
        entropy = 0.0
    return parentmass, int(values.size), entropy


def quantile_bins(series: pd.Series, bins: int = 4) -> pd.Series:
    ranked = series.rank(method="first", pct=True)
    values = np.minimum((ranked * bins).astype(int), bins - 1)
    return values.astype(str)


def build_population_frame(
    *,
    metadata_csv: Path,
    labels_tsv: Path,
    fingerprints_npz: Path,
    spectrum_dir: Path,
    expected_population_size: int | None,
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_csv)
    required_metadata = {"spec_name", "inchi_key_first_block"}
    missing = required_metadata.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata lacks columns: {sorted(missing)}")
    if metadata["spec_name"].astype(str).duplicated().any():
        raise ValueError("Population metadata has duplicate spec_name values")
    if (
        expected_population_size is not None
        and len(metadata) != expected_population_size
    ):
        raise ValueError(
            f"Expected {expected_population_size} population rows, found {len(metadata)}"
        )

    labels = pd.read_csv(labels_tsv, sep="\t")
    required_labels = {
        "spec",
        "formula",
        "inchikey",
        "instrument",
        "ionization",
    }
    missing = required_labels.difference(labels.columns)
    if missing:
        raise ValueError(f"Labels lack columns: {sorted(missing)}")
    if labels["spec"].astype(str).duplicated().any():
        raise ValueError("Labels have duplicate spec values")

    metadata = metadata.copy()
    metadata["spec_name"] = metadata["spec_name"].astype(str)
    labels = labels.copy()
    labels["spec"] = labels["spec"].astype(str)
    frame = metadata.merge(
        labels[list(required_labels)],
        left_on="spec_name",
        right_on="spec",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_labels = frame.loc[frame["_merge"] != "both", "spec_name"].tolist()
    if missing_labels:
        raise ValueError(f"Population rows lack labels: {missing_labels[:5]}")
    frame = frame.drop(columns=["_merge", "spec"])
    frame["inchi_key_first_block"] = frame["inchi_key_first_block"].map(
        normalize_inchi_block
    )
    label_blocks = frame["inchikey"].map(normalize_inchi_block)
    if not frame["inchi_key_first_block"].equals(label_blocks):
        raise ValueError("Metadata and labels disagree on InChIKey connectivity")
    if (frame["inchi_key_first_block"] == "").any():
        raise ValueError("Population contains an empty InChIKey connectivity block")

    with np.load(fingerprints_npz, mmap_mode="r") as loaded:
        if "mist_binary" not in loaded.files:
            raise ValueError("Fingerprint archive lacks mist_binary")
        fingerprints = np.asarray(loaded["mist_binary"])
    if fingerprints.ndim != 2 or fingerprints.shape[0] != len(frame):
        raise ValueError(
            "Fingerprint rows do not align with population metadata: "
            f"{fingerprints.shape} vs {len(frame)}"
        )
    if "fingerprint_index" in frame:
        expected_indices = np.arange(len(frame))
        actual_indices = frame["fingerprint_index"].to_numpy(dtype=int)
        if not np.array_equal(actual_indices, expected_indices):
            raise ValueError("Metadata fingerprint_index is not contiguous and ordered")
    frame["mist_active_bits"] = fingerprints.sum(axis=1).astype(float)

    spectrum_features = []
    missing_spectra = []
    for spec_name in frame["spec_name"]:
        path = spectrum_dir / f"{spec_name}.ms"
        if not path.is_file():
            missing_spectra.append(spec_name)
            continue
        spectrum_features.append(read_spectrum_features(path))
    if missing_spectra:
        raise ValueError(f"Population rows lack spectrum files: {missing_spectra[:5]}")
    frame[["parentmass", "peak_count", "spectral_entropy"]] = pd.DataFrame(
        spectrum_features,
        index=frame.index,
    )

    formula_rows = pd.DataFrame(
        [formula_features(value) for value in frame["formula"]],
        index=frame.index,
    )
    frame = pd.concat([frame, formula_rows], axis=1)
    replicate_counts = frame["inchi_key_first_block"].value_counts()
    frame["replicate_count"] = frame["inchi_key_first_block"].map(replicate_counts)
    frame["instrument_group"] = frame["instrument"].fillna("unknown").astype(str)
    frame["ionization_group"] = frame["ionization"].fillna("unknown").astype(str)
    for column in (
        "parentmass",
        "peak_count",
        "spectral_entropy",
        "mist_active_bits",
    ):
        frame[f"{column}_bin"] = quantile_bins(frame[column])
    frame["replicate_count_bin"] = pd.cut(
        frame["replicate_count"],
        bins=[0, 1, 4, 16, 64, 128, math.inf],
        labels=["1", "2-4", "5-16", "17-64", "65-128", "129+"],
    ).astype(str)
    return frame


def make_balance_matrix(
    target: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    blocks = []
    target_props = []
    weights = []
    feature_names = []
    for column in BALANCE_COLUMNS:
        categories = sorted(
            set(target[column].astype(str)).union(candidates[column].astype(str))
        )
        candidate_block = np.column_stack(
            [
                (candidates[column].astype(str) == value).to_numpy(float)
                for value in categories
            ]
        )
        target_block = np.asarray(
            [(target[column].astype(str) == value).mean() for value in categories],
            dtype=float,
        )
        variance = np.maximum(target_block * (1.0 - target_block), 0.02)
        block_weights = (1.0 / len(categories)) / variance
        blocks.append(candidate_block)
        target_props.extend(target_block)
        weights.extend(block_weights)
        feature_names.extend(f"{column}={value}" for value in categories)
    return (
        np.concatenate(blocks, axis=1),
        np.asarray(target_props),
        np.asarray(weights),
        feature_names,
    )


def select_balanced_sequence(
    target: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    size: int,
    namespace: str,
) -> pd.DataFrame:
    if size <= 0 or size > len(candidates):
        raise ValueError(f"Invalid selection size {size} for {len(candidates)} rows")
    candidates = candidates.sort_values("spec_name", kind="stable").reset_index(
        drop=True
    )
    matrix, target_props, weights, _ = make_balance_matrix(target, candidates)
    selected_mask = np.zeros(len(candidates), dtype=bool)
    counts = np.zeros(matrix.shape[1], dtype=float)
    row_norm = (matrix * matrix * weights).sum(axis=1)
    jitter = np.asarray(
        [
            int(stable_hash(namespace, name)[:15], 16) / float(16**15)
            for name in candidates["spec_name"]
        ]
    )
    selected_indices = []
    for step in range(1, size + 1):
        residual = counts - step * target_props
        scores = row_norm + matrix @ (2.0 * weights * residual)
        scores = scores + jitter * 1e-9
        scores[selected_mask] = np.inf
        chosen = int(np.argmin(scores))
        selected_mask[chosen] = True
        selected_indices.append(chosen)
        counts += matrix[chosen]
    selected = candidates.iloc[selected_indices].copy().reset_index(drop=True)
    selected["selection_order"] = np.arange(1, len(selected) + 1)
    return selected


def representative_per_molecule(frame: pd.DataFrame, namespace: str) -> pd.DataFrame:
    ranked = frame.copy()
    ranked["representative_hash"] = [
        stable_hash(namespace, block, spec)
        for block, spec in zip(
            ranked["inchi_key_first_block"], ranked["spec_name"], strict=True
        )
    ]
    return (
        ranked.sort_values(
            ["inchi_key_first_block", "representative_hash"], kind="stable"
        )
        .drop_duplicates("inchi_key_first_block", keep="first")
        .drop(columns="representative_hash")
        .reset_index(drop=True)
    )


def panel_profile(frame: pd.DataFrame) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "rows": len(frame),
        "unique_specs": int(frame["spec_name"].nunique()),
        "unique_molecules": int(frame["inchi_key_first_block"].nunique()),
    }
    profile["continuous"] = {}
    for column in CONTINUOUS_FEATURES:
        values = frame[column].astype(float)
        profile["continuous"][column] = {
            "mean": float(values.mean()),
            "q10": float(values.quantile(0.1)),
            "q50": float(values.quantile(0.5)),
            "q90": float(values.quantile(0.9)),
        }
    profile["categorical"] = {}
    for column in ("instrument_group", "ionization_group"):
        profile["categorical"][column] = {
            str(key): float(value)
            for key, value in frame[column]
            .value_counts(normalize=True)
            .sort_index()
            .items()
        }
    return profile


def profile_discrepancy(target: pd.DataFrame, panel: pd.DataFrame) -> dict[str, Any]:
    smd = {}
    for column in CONTINUOUS_FEATURES:
        denominator = float(target[column].astype(float).std(ddof=0))
        difference = float(panel[column].mean() - target[column].mean())
        smd[column] = abs(difference / denominator) if denominator > 0 else 0.0
    category_differences = {}
    for column in ("instrument_group", "ionization_group"):
        target_shares = target[column].astype(str).value_counts(normalize=True)
        panel_shares = panel[column].astype(str).value_counts(normalize=True)
        categories = sorted(set(target_shares.index).union(panel_shares.index))
        category_differences[column] = max(
            abs(float(target_shares.get(value, 0.0) - panel_shares.get(value, 0.0)))
            for value in categories
        )
    return {
        "absolute_standardized_mean_difference": smd,
        "max_categorical_share_difference": category_differences,
        "max_absolute_smd": max(smd.values()),
        "max_absolute_category_share_difference": max(category_differences.values()),
    }


def write_panel(frame: pd.DataFrame, path: Path, panel_name: str) -> dict[str, Any]:
    output = pd.DataFrame(
        {
            "spec_name": frame["spec_name"],
            "inchikey_first_block": frame["inchi_key_first_block"],
            "formula": frame["formula"],
            "selection_hash": [
                stable_hash(SELECTION_NAMESPACE, panel_name, spec)
                for spec in frame["spec_name"]
            ],
            "panel": panel_name,
            "selection_order": np.arange(1, len(frame) + 1),
        }
    )
    output.to_csv(path, sep="\t", index=False)
    ordered_hash = hashlib.sha256(
        "".join(f"{name}\n" for name in output["spec_name"]).encode("utf-8")
    ).hexdigest()
    return {
        "path": str(path),
        "rows": len(output),
        "sha256": sha256_file(path),
        "ordered_spec_names_sha256": ordered_hash,
    }


def build_compact_benchmark(
    *,
    metadata_csv: Path,
    labels_tsv: Path,
    fingerprints_npz: Path,
    spectrum_dir: Path,
    exclusion_manifests: list[Path],
    output_dir: Path,
    expected_population_size: int | None = 17082,
    micro_sizes: tuple[int, ...] = (128, 256, 512),
    macro_size: int = 64,
) -> dict[str, Any]:
    if tuple(sorted(set(micro_sizes))) != micro_sizes:
        raise ValueError("micro_sizes must be strictly increasing")
    population = build_population_frame(
        metadata_csv=metadata_csv,
        labels_tsv=labels_tsv,
        fingerprints_npz=fingerprints_npz,
        spectrum_dir=spectrum_dir,
        expected_population_size=expected_population_size,
    )
    excluded_specs = load_exclusion_manifests(exclusion_manifests)
    unknown_exclusions = excluded_specs.difference(population["spec_name"])
    if unknown_exclusions:
        raise ValueError(
            f"Exclusion specs are absent from population: {sorted(unknown_exclusions)[:5]}"
        )
    micro_candidates = population[~population["spec_name"].isin(excluded_specs)].copy()
    micro = select_balanced_sequence(
        population,
        micro_candidates,
        size=micro_sizes[-1],
        namespace=f"{SELECTION_NAMESPACE}:micro",
    )

    excluded_blocks = set(
        population.loc[
            population["spec_name"].isin(excluded_specs), "inchi_key_first_block"
        ]
    )
    excluded_blocks.update(micro["inchi_key_first_block"])
    macro_population = representative_per_molecule(
        population,
        namespace=f"{SELECTION_NAMESPACE}:macro-target",
    )
    macro_candidates = representative_per_molecule(
        population[
            ~population["inchi_key_first_block"].isin(excluded_blocks)
            & ~population["spec_name"].isin(excluded_specs)
        ],
        namespace=f"{SELECTION_NAMESPACE}:macro-candidates",
    )
    macro = select_balanced_sequence(
        macro_population,
        macro_candidates,
        size=macro_size,
        namespace=f"{SELECTION_NAMESPACE}:macro",
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for size in micro_sizes:
        name = f"msg_compact_micro{size}_v1"
        outputs[name] = write_panel(micro.iloc[:size], output_dir / f"{name}.tsv", name)
    outputs["msg_compact_macro64_v1"] = write_panel(
        macro,
        output_dir / "msg_compact_macro64_v1.tsv",
        "msg_compact_macro64_v1",
    )

    micro_names = set(micro["spec_name"])
    macro_names = set(macro["spec_name"])
    macro_blocks = set(macro["inchi_key_first_block"])
    if micro_names.intersection(macro_names):
        raise AssertionError("Micro and macro panels overlap by spec_name")
    if macro_blocks.intersection(excluded_blocks):
        raise AssertionError("Macro panel overlaps prior or micro molecules")

    report = {
        "schema_version": SCHEMA_VERSION,
        "selection_namespace": SELECTION_NAMESPACE,
        "inputs": {
            "metadata_csv": {
                "path": str(metadata_csv),
                "sha256": sha256_file(metadata_csv),
            },
            "labels_tsv": {"path": str(labels_tsv), "sha256": sha256_file(labels_tsv)},
            "fingerprints_npz": {
                "path": str(fingerprints_npz),
                "sha256": sha256_file(fingerprints_npz),
            },
            "spectrum_dir": str(spectrum_dir),
            "exclusion_manifests": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in exclusion_manifests
            ],
        },
        "counts": {
            "population_rows": len(population),
            "population_unique_molecules": int(
                population["inchi_key_first_block"].nunique()
            ),
            "excluded_specs": len(excluded_specs),
            "micro_candidate_rows": len(micro_candidates),
            "macro_candidate_molecules": len(macro_candidates),
        },
        "quality_checks": {
            "population_spec_unique": True,
            "labels_join_coverage": 1.0,
            "spectrum_file_coverage": 1.0,
            "fingerprint_alignment": True,
            "micro_panels_nested": True,
            "micro_macro_spec_overlap": 0,
            "macro_prior_or_micro_molecule_overlap": 0,
            "target_fields_used_for_model_scoring": [],
        },
        "profiles": {
            "population_micro": panel_profile(population),
            "population_macro": panel_profile(macro_population),
            **{
                f"micro{size}": panel_profile(micro.iloc[:size]) for size in micro_sizes
            },
            "macro64": panel_profile(macro),
        },
        "distribution_checks": {
            **{
                f"micro{size}_vs_population": profile_discrepancy(
                    population, micro.iloc[:size]
                )
                for size in micro_sizes
            },
            "macro64_vs_unique_molecule_population": profile_discrepancy(
                macro_population, macro
            ),
        },
        "outputs": outputs,
    }
    report_path = output_dir / "msg_compact_selection_report_v1.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--labels-tsv", required=True)
    parser.add_argument("--fingerprints-npz", required=True)
    parser.add_argument("--spectrum-dir", required=True)
    parser.add_argument("--exclude-manifest", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-population-size", type=int, default=17082)
    parser.add_argument("--micro-sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--macro-size", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_compact_benchmark(
        metadata_csv=Path(args.metadata_csv).expanduser().resolve(),
        labels_tsv=Path(args.labels_tsv).expanduser().resolve(),
        fingerprints_npz=Path(args.fingerprints_npz).expanduser().resolve(),
        spectrum_dir=Path(args.spectrum_dir).expanduser().resolve(),
        exclusion_manifests=[
            Path(value).expanduser().resolve() for value in args.exclude_manifest
        ],
        output_dir=Path(args.output_dir).expanduser().resolve(),
        expected_population_size=args.expected_population_size,
        micro_sizes=tuple(args.micro_sizes),
        macro_size=args.macro_size,
    )
    print(json.dumps(report["outputs"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
