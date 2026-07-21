#!/usr/bin/env python3
"""Build formula-blind MIST inputs from mass-consistent MIST-CF predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from rdkit import Chem


PRECURSOR_PPM_TOLERANCE = 10.0
FORMULA_SOURCE = (
    "Highest-scoring SIRIUS-consistent MIST-CF candidate within 10 ppm; "
    "no ground-truth formula"
)
ELECTRON_MASS = 0.00054858
PERIODIC_TABLE = Chem.GetPeriodicTable()
ION_REMAP = {
    "[M+NH4]+": "[M+H3N+H]+",
    "[M-2H2O+H]+": "[M-H4O2+H]+",
}


def element_mass(symbol: str) -> float:
    return float(PERIODIC_TABLE.GetMostCommonIsotopeMass(symbol))


ION_TO_MASS = {
    "[M+H]+": element_mass("H") - ELECTRON_MASS,
    "[M+Na]+": element_mass("Na") - ELECTRON_MASS,
    "[M+K]+": element_mass("K") - ELECTRON_MASS,
    "[M-H2O+H]+": -element_mass("O") - element_mass("H") - ELECTRON_MASS,
    "[M+H3N+H]+": element_mass("N") + 4 * element_mass("H") - ELECTRON_MASS,
    "[M]+": -ELECTRON_MASS,
    "[M-H4O2+H]+": -2 * element_mass("O") - 3 * element_mass("H") - ELECTRON_MASS,
}


def formula_mass(formula: str) -> float:
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    if not parts or "".join(f"{element}{count}" for element, count in parts) != formula:
        raise ValueError(f"Unsupported molecular formula {formula!r}")
    return sum(element_mass(element) * int(count or "1") for element, count in parts)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ranked_predictions(path: Path) -> dict[str, list[dict[str, str]]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"spec", "cand_form", "cand_ion", "scores", "parentmasses"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"MIST-CF output is missing columns {sorted(required)}")
    ranked: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        spectrum_id = row["spec"].strip()
        if not spectrum_id:
            raise ValueError("MIST-CF output contains an empty spectrum ID")
        float(row["scores"])
        ranked.setdefault(spectrum_id, []).append(row)
    for candidates in ranked.values():
        candidates.sort(key=lambda row: float(row["scores"]), reverse=True)
    return ranked


def select_mass_consistent_candidate(
    candidates: list[dict[str, str]],
    observed_precursor_mz: float,
    minimum_rank: int = 1,
) -> tuple[dict[str, str], int, float, float]:
    for rank, row in enumerate(candidates, start=1):
        if rank < minimum_rank:
            continue
        ion = ION_REMAP.get(row["cand_ion"].strip(), row["cand_ion"].strip())
        if ion not in ION_TO_MASS:
            continue
        theoretical_mz = formula_mass(row["cand_form"].strip()) + ION_TO_MASS[ion]
        ppm_error = abs(theoretical_mz - observed_precursor_mz) / observed_precursor_mz * 1e6
        if ppm_error <= PRECURSOR_PPM_TOLERANCE:
            selected = dict(row)
            selected["cand_ion"] = ion
            return selected, rank, theoretical_mz, ppm_error
    raise ValueError(
        f"No MIST-CF candidate is mass-consistent within "
        f"{PRECURSOR_PPM_TOLERANCE} ppm of {observed_precursor_mz}"
    )


def read_mgf(path: Path) -> list[tuple[str, dict[str, str], list[str]]]:
    blocks: list[tuple[str, dict[str, str], list[str]]] = []
    current: list[str] | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if line == "BEGIN IONS":
            if current is not None:
                raise ValueError("Nested BEGIN IONS in MGF")
            current = []
        elif line == "END IONS":
            if current is None:
                raise ValueError("END IONS without BEGIN IONS")
            headers = {}
            for value in current:
                if "=" in value:
                    key, field = value.split("=", 1)
                    headers[key.upper()] = field.strip()
            spectrum_id = (
                headers.get("SCANS")
                or headers.get("FEATURE_ID")
                or headers.get("TITLE")
            )
            if not spectrum_id:
                raise ValueError("MGF block lacks SCANS, FEATURE_ID, and TITLE")
            blocks.append((spectrum_id, headers, current))
            current = None
        elif current is not None and line:
            current.append(line)
    if current is not None:
        raise ValueError("Unterminated MGF block")
    ids = [spectrum_id for spectrum_id, _, _ in blocks]
    if len(ids) != len(set(ids)):
        raise ValueError("MGF contains duplicate spectrum IDs")
    return blocks


def peak_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if "=" not in line]


def build_dataset(
    mgf: Path,
    predictions: Path,
    output_dir: Path,
    minimum_candidate_ranks: dict[str, int] | None = None,
    *,
    formula_source: str = FORMULA_SOURCE,
    manifest_kind: str = "Mass-consistent MIST-CF predicted-formula bridge into official MIST",
) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_dir = output_dir / "spec_files"
    spec_dir.mkdir()

    blocks = read_mgf(mgf)
    ranked = read_ranked_predictions(predictions)
    mgf_ids = [spectrum_id for spectrum_id, _, _ in blocks]
    missing = sorted(set(mgf_ids) - set(ranked))
    extra = sorted(set(ranked) - set(mgf_ids))
    if missing or extra:
        raise ValueError(
            f"MIST-CF/MGF ID mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )

    forced_blocks: list[str] = []
    labels: list[dict[str, str]] = []
    for spectrum_id, headers, lines in blocks:
        observed_precursor_mz = float(headers["PEPMASS"].split()[0])
        minimum_rank = (minimum_candidate_ranks or {}).get(spectrum_id, 1)
        row, candidate_rank, theoretical_mz, ppm_error = select_mass_consistent_candidate(
            ranked[spectrum_id], observed_precursor_mz, minimum_rank
        )
        formula = row["cand_form"].strip()
        ion = row["cand_ion"].strip()
        parentmass = headers["PEPMASS"].split()[0]
        peaks = peak_lines(lines)
        if not formula or not ion or not peaks:
            raise ValueError(f"Incomplete predicted-formula input for {spectrum_id}")
        forced_headers = [
            f"TITLE={spectrum_id}",
            f"SCANS={spectrum_id}",
            f"FEATURE_ID={spectrum_id}",
            f"PEPMASS={parentmass}",
            f"FORMULA={formula}",
            f"IONIZATION={ion}",
            f"ADDUCT={ion}",
        ]
        if "INSTRUMENT" in headers:
            forced_headers.append(f"INSTRUMENT={headers['INSTRUMENT']}")
        forced_blocks.append(
            "\n".join(["BEGIN IONS", *forced_headers, *peaks, "END IONS"])
        )
        ms_lines = [
            f">compound {spectrum_id}",
            f">parentmass {parentmass}",
            f">formula {formula}",
            f">ionization {ion}",
            ">ms2",
            *peaks,
            "",
        ]
        (spec_dir / f"{spectrum_id}.ms").write_text("\n".join(ms_lines))
        labels.append(
            {
                "dataset": output_dir.name,
                "spec": spectrum_id,
                "formula": formula,
                "ionization": ion,
                "parentmass": parentmass,
                "predicted_parentmass": f"{theoretical_mz:.12g}",
                "candidate_rank": str(candidate_rank),
                "precursor_ppm_error": f"{ppm_error:.12g}",
            }
        )

    forced_mgf = output_dir / "forced_formula.mgf"
    forced_mgf.write_text("\n\n".join(forced_blocks) + "\n")
    labels_path = output_dir / "labels.tsv"
    with labels_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(labels[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(labels)
    manifest = {
        "schema_version": 1,
        "kind": manifest_kind,
        "formula_source": formula_source,
        "precursor_ppm_tolerance": PRECURSOR_PPM_TOLERANCE,
        "fallback_rows": sum(int(row["candidate_rank"]) > 1 for row in labels),
        "maximum_candidate_rank": max(int(row["candidate_rank"]) for row in labels),
        "rows": len(labels),
        "mgf": str(mgf.resolve()),
        "mgf_sha256": sha256_file(mgf),
        "mist_cf_predictions": str(predictions.resolve()),
        "mist_cf_predictions_sha256": sha256_file(predictions),
        "forced_mgf_sha256": sha256_file(forced_mgf),
        "labels_sha256": sha256_file(labels_path),
    }
    (output_dir / "formula_bridge_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mgf", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-candidate-ranks", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    minimum_candidate_ranks = None
    if args.minimum_candidate_ranks is not None:
        payload = json.loads(args.minimum_candidate_ranks.read_text())
        minimum_candidate_ranks = {
            str(key): int(value)
            for key, value in payload.get("minimum_candidate_ranks", payload).items()
        }
        if any(rank < 1 for rank in minimum_candidate_ranks.values()):
            raise ValueError("Minimum candidate ranks must be positive")
    print(
        json.dumps(
            build_dataset(
                args.mgf,
                args.predictions,
                args.output_dir,
                minimum_candidate_ranks,
            )
        )
    )


if __name__ == "__main__":
    main()
