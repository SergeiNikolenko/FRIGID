#!/usr/bin/env python3
"""Unpack SIRIUS 5 project-space trees into the layout expected by MIST."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path

from scripts.prepare_mist_predicted_formula_dataset import ION_TO_MASS, formula_mass


PRECURSOR_PPM_TOLERANCE = 10.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _formula_with_hydrogen_delta(formula: str, delta: int) -> str:
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    if not parts or "".join(f"{element}{count}" for element, count in parts) != formula:
        raise ValueError(f"Unsupported molecular formula {formula!r}")
    output = []
    found_hydrogen = False
    for element, count_text in parts:
        count = int(count_text or "1")
        if element == "H":
            count += delta
            found_hydrogen = True
        if count < 0:
            raise ValueError(f"Invalid hydrogen normalization for {formula!r}")
        if count:
            output.append(element if count == 1 else f"{element}{count}")
    if not found_hydrogen and delta:
        raise ValueError(f"Cannot normalize hydrogen-free formula {formula!r}")
    return "".join(output)


def _expected_tree_formula(formula: str, adduct: str) -> tuple[str, str]:
    if adduct == "[M]+":
        return _formula_with_hydrogen_delta(formula, -1), "sirius_[M]+_minus_H"
    return formula, "identity"


def _read_ms_headers(path: Path) -> dict[str, str]:
    headers = {}
    for line in path.read_text().splitlines():
        if line.startswith(">") and " " in line:
            key, value = line[1:].split(" ", 1)
            headers[key] = value.strip()
    return headers


def unpack_project(project_dir: Path, labels_path: Path) -> list[dict[str, str]]:
    with labels_path.open(newline="") as handle:
        expected = {
            row["spec"]: {
                "formula": row["formula"],
                "ionization": row["ionization"].replace(" ", ""),
            }
            for row in csv.DictReader(handle, delimiter="\t")
        }
    rows: list[dict[str, str]] = []
    for compound_dir in sorted(path for path in project_dir.iterdir() if path.is_dir()):
        info = {}
        for line in (compound_dir / "compound.info").read_text().splitlines():
            if "\t" in line:
                key, value = line.split("\t", 1)
                info[key] = value
        spectrum_id = info["name"]
        if spectrum_id not in expected:
            raise ValueError(f"Unexpected SIRIUS spectrum ID {spectrum_id}")
        expected_formula = expected[spectrum_id]["formula"]
        expected_adduct = expected[spectrum_id]["ionization"]
        if info["ionType"].replace(" ", "") != expected_adduct:
            raise ValueError(
                f"SIRIUS compound adduct mismatch for {spectrum_id}: "
                f"expected={expected_adduct} observed={info['ionType']}"
            )
        ms_headers = _read_ms_headers(compound_dir / "spectrum.ms")
        if ms_headers.get("formula") != expected_formula:
            raise ValueError(
                f"SIRIUS input formula mismatch for {spectrum_id}: "
                f"expected={expected_formula} observed={ms_headers.get('formula')}"
            )
        if ms_headers.get("ionization", "").replace(" ", "") != expected_adduct:
            raise ValueError(
                f"SIRIUS input adduct mismatch for {spectrum_id}: "
                f"expected={expected_adduct} observed={ms_headers.get('ionization')}"
            )
        for name in ("scores", "spectra", "trees"):
            archive = compound_dir / name
            extracted = compound_dir / f"{name}.unpacked"
            if archive.is_file():
                with zipfile.ZipFile(archive) as bundle:
                    bundle.extractall(extracted)
                archive.replace(compound_dir / f"{name}.zip")
                extracted.replace(archive)
            elif not archive.is_dir():
                raise FileNotFoundError(f"Missing SIRIUS {name} for {spectrum_id}")
        trees = list((compound_dir / "trees").glob("*.json"))
        spectra = list((compound_dir / "spectra").glob("*.tsv"))
        scores = list((compound_dir / "scores").glob("*.info"))
        if len(trees) != 1 or len(spectra) != 1 or len(scores) != 1:
            raise ValueError(f"Expected one forced-formula tree for {spectrum_id}")
        tree = json.loads(trees[0].read_text())
        annotations = tree["annotations"]
        observed_formula = tree["molecularFormula"]
        observed_adduct = annotations["precursorIonType"].replace(" ", "")
        expected_tree_formula, formula_normalization = _expected_tree_formula(
            expected_formula, expected_adduct
        )
        if observed_formula != expected_tree_formula:
            raise ValueError(
                f"SIRIUS formula mismatch for {spectrum_id}: "
                f"expected={expected_tree_formula} "
                f"observed={observed_formula}"
            )
        if observed_adduct != expected_adduct:
            raise ValueError(
                f"SIRIUS adduct mismatch for {spectrum_id}: "
                f"expected={expected_adduct} "
                f"observed={observed_adduct}"
            )
        mist_adduct = (
            "[M+H]+"
            if formula_normalization == "sirius_[M]+_minus_H"
            else observed_adduct
        )
        mist_precursor_mz = formula_mass(observed_formula) + ION_TO_MASS[mist_adduct]
        sirius_precursor_mz = float(info["ionMass"])
        mist_precursor_ppm_error = (
            abs(mist_precursor_mz - sirius_precursor_mz)
            / sirius_precursor_mz
            * 1e6
        )
        if mist_precursor_ppm_error > PRECURSOR_PPM_TOLERANCE:
            raise ValueError(
                f"MIST label mass mismatch for {spectrum_id}: "
                f"formula={observed_formula} adduct={mist_adduct} "
                f"ppm={mist_precursor_ppm_error}"
            )
        rows.append(
            {
                "spec_name": spectrum_id,
                "spec_file": str(spectra[0].resolve()),
                "tree_file": str(trees[0].resolve()),
                "adduct": observed_adduct,
                "pred_formula": observed_formula,
                "mist_cf_formula": expected_formula,
                "tree_formula": observed_formula,
                "formula_normalization": formula_normalization,
                "mist_adduct": mist_adduct,
                "mist_precursor_ppm_error": f"{mist_precursor_ppm_error:.12g}",
                "parentmass": info["ionMass"],
            }
        )
    observed = {row["spec_name"] for row in rows}
    if observed != set(expected):
        raise ValueError(
            f"SIRIUS/labels ID mismatch: missing={sorted(set(expected)-observed)[:5]}, "
            f"extra={sorted(observed-set(expected))[:5]}"
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mist-labels-output", type=Path, required=True)
    parser.add_argument("--formula-manifest", type=Path, required=True)
    parser.add_argument("--sirius-audit", type=Path, required=True)
    parser.add_argument("--bridge-manifest-output", type=Path, required=True)
    return parser.parse_args()


def write_summary(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "",
                "spec_name",
                "spec_file",
                "tree_file",
                "adduct",
                "pred_formula",
                "mist_cf_formula",
                "tree_formula",
                "formula_normalization",
                "mist_adduct",
                "mist_precursor_ppm_error",
                "parentmass",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows({"": index, **row} for index, row in enumerate(rows))


def write_mist_labels(rows: list[dict[str, str]], output: Path) -> None:
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "spec",
                "formula",
                "ionization",
                "parentmass",
                "mist_cf_formula",
                "formula_normalization",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(
            {
                "dataset": output.parent.name,
                "spec": row["spec_name"],
                "formula": row["tree_formula"],
                "ionization": row["mist_adduct"],
                "parentmass": row["parentmass"],
                "mist_cf_formula": row["mist_cf_formula"],
                "formula_normalization": row["formula_normalization"],
            }
            for row in rows
        )


def write_bridge_manifest(
    rows: list[dict[str, str]],
    formula_manifest: Path,
    sirius_audit: Path,
    labels: Path,
    summary: Path,
    mist_labels: Path,
    output: Path,
) -> dict:
    formula_payload = json.loads(formula_manifest.read_text())
    audit_payload = json.loads(sirius_audit.read_text())
    if not formula_payload.get("sirius_consistency_validated"):
        raise ValueError("Formula manifest lacks successful SIRIUS consistency audit")
    if formula_payload.get("sirius_consistency_audit_sha256") != sha256_file(
        sirius_audit
    ):
        raise ValueError("Formula manifest is not bound to the supplied SIRIUS audit")
    if audit_payload.get("mismatch_count") != 0:
        raise ValueError("SIRIUS bridge still contains formula/adduct mismatches")
    if audit_payload.get("rows") != len(rows):
        raise ValueError("SIRIUS audit row count does not match unpacked bridge")
    labels_sha256 = sha256_file(labels)
    if (
        formula_payload.get("labels_sha256") != labels_sha256
        or audit_payload.get("labels_sha256") != labels_sha256
    ):
        raise ValueError("Formula manifest and SIRIUS audit are not bound to labels")
    per_id_digest = hashlib.sha256(
        json.dumps(
            [
                {
                    "spec": row["spec_name"],
                    "mist_cf_formula": row["mist_cf_formula"],
                    "tree_formula": row["tree_formula"],
                    "sirius_adduct": row["adduct"],
                    "mist_adduct": row["mist_adduct"],
                    "formula_normalization": row["formula_normalization"],
                    "mist_precursor_ppm_error": row["mist_precursor_ppm_error"],
                }
                for row in rows
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "kind": "SIRIUS-validated formula-blind bridge into official MIST",
        "rows": len(rows),
        "formula_source": formula_payload["formula_source"],
        "precursor_ppm_tolerance": PRECURSOR_PPM_TOLERANCE,
        "fallback_rows": formula_payload["fallback_rows"],
        "maximum_candidate_rank": formula_payload["maximum_candidate_rank"],
        "formula_manifest_sha256": sha256_file(formula_manifest),
        "sirius_audit_sha256": sha256_file(sirius_audit),
        "summary_sha256": sha256_file(summary),
        "mist_labels_sha256": sha256_file(mist_labels),
        "sirius_tree_evidence_sha256": audit_payload[
            "sirius_tree_evidence_sha256"
        ],
        "per_id_mapping_sha256": per_id_digest,
        "maximum_mist_precursor_ppm_error": max(
            float(row["mist_precursor_ppm_error"]) for row in rows
        ),
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    args = parse_args()
    rows = unpack_project(args.project_dir, args.labels)
    write_summary(rows, args.output)
    write_mist_labels(rows, args.mist_labels_output)
    write_bridge_manifest(
        rows,
        args.formula_manifest,
        args.sirius_audit,
        args.labels,
        args.output,
        args.mist_labels_output,
        args.bridge_manifest_output,
    )
    print(f"Prepared {len(rows)} SIRIUS trees for MIST")


if __name__ == "__main__":
    main()
