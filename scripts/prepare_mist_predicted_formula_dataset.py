#!/usr/bin/env python3
"""Build formula-blind MIST inputs from MIST-CF top-1 predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_top_predictions(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    required = {"spec", "cand_form", "cand_ion", "scores", "parentmasses"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"MIST-CF output is missing columns {sorted(required)}")
    top: dict[str, dict[str, str]] = {}
    for row in rows:
        spectrum_id = row["spec"].strip()
        if not spectrum_id:
            raise ValueError("MIST-CF output contains an empty spectrum ID")
        score = float(row["scores"])
        if spectrum_id not in top or score > float(top[spectrum_id]["scores"]):
            top[spectrum_id] = row
    return top


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


def build_dataset(mgf: Path, predictions: Path, output_dir: Path) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_dir = output_dir / "spec_files"
    spec_dir.mkdir()

    blocks = read_mgf(mgf)
    top = read_top_predictions(predictions)
    mgf_ids = [spectrum_id for spectrum_id, _, _ in blocks]
    missing = sorted(set(mgf_ids) - set(top))
    extra = sorted(set(top) - set(mgf_ids))
    if missing or extra:
        raise ValueError(
            f"MIST-CF/MGF ID mismatch: missing={missing[:5]}, extra={extra[:5]}"
        )

    forced_blocks: list[str] = []
    labels: list[dict[str, str]] = []
    for spectrum_id, headers, lines in blocks:
        row = top[spectrum_id]
        formula = row["cand_form"].strip()
        ion = row["cand_ion"].strip()
        parentmass = headers.get("PEPMASS", row["parentmasses"]).split()[0]
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
        "kind": "MIST-CF top-1 predicted-formula bridge into official MIST",
        "formula_source": "MIST-CF top-1 prediction; no ground-truth formula",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build_dataset(args.mgf, args.predictions, args.output_dir)))


if __name__ == "__main__":
    main()
