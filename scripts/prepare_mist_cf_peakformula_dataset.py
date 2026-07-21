#!/usr/bin/env python3
"""Build official-MIST PeakFormula inputs from MIST-CF top-1 subformulae."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

from scripts.prepare_mist_predicted_formula_dataset import (
    PRECURSOR_PPM_TOLERANCE,
    build_dataset,
    sha256_file,
)


FORMULA_SOURCE = (
    "Top-1 formula from connectivity-clean MIST-CF used only for "
    "peak-to-subformula features; no ground-truth formula"
)
FORMULA_MANIFEST_KIND = "MIST-CF top-1 predicted-formula inputs for official MIST"
FEATURE_BRIDGE_KIND = "MIST-CF top-1 peak-to-subformula bridge into official MIST"
SUBFORMULA_PPM_TOLERANCE = 15.0
OFFICIAL_MIST_ELEMENTS = {
    "C",
    "N",
    "P",
    "O",
    "S",
    "Si",
    "I",
    "H",
    "Cl",
    "F",
    "Br",
    "B",
    "Se",
    "Fe",
    "Co",
    "As",
}
FORMULA_PATTERN = re.compile(r"([A-Z][a-z]*)([0-9]*)")


def _json_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _official_mist_formula_counts(formula: str) -> dict[str, int] | None:
    matches = list(FORMULA_PATTERN.finditer(formula))
    if not matches or "".join(match.group(0) for match in matches) != formula:
        return None
    counts: dict[str, int] = {}
    for match in matches:
        element = match.group(1)
        if element not in OFFICIAL_MIST_ELEMENTS:
            return None
        count = int(match.group(2) or 1)
        if count <= 0 or element in counts:
            return None
        counts[element] = count
    return counts


def _validate_fragment_formula(formula: str, root_counts: dict[str, int]) -> None:
    counts = _official_mist_formula_counts(formula)
    if counts is None:
        raise ValueError(f"Official MIST cannot parse subformula {formula!r}")
    if any(count > root_counts.get(element, 0) for element, count in counts.items()):
        raise ValueError(f"MIST-CF fragment {formula!r} is not a root subformula")


def build_peakformula_dataset(
    mgf: Path,
    predictions: Path,
    subformula_dir: Path,
    output_dir: Path,
    mist_cf_git_commit: str,
    mist_cf_checkpoint_sha256: str,
) -> dict:
    formula_manifest = build_dataset(
        mgf,
        predictions,
        output_dir,
        formula_source=FORMULA_SOURCE,
        manifest_kind=FORMULA_MANIFEST_KIND,
    )
    if (
        formula_manifest["fallback_rows"] != 0
        or formula_manifest["maximum_candidate_rank"] != 1
    ):
        raise ValueError("Paper-aligned MIST lane requires the MIST-CF top-1 candidate")

    labels_path = output_dir / "labels.tsv"
    with labels_path.open(newline="") as handle:
        labels = list(csv.DictReader(handle, delimiter="\t"))
    trees_dir = output_dir / "peakformula_trees"
    trees_dir.mkdir()
    summary_dir = output_dir / "sirius_outputs" / "summary_statistics"
    summary_dir.mkdir(parents=True)

    summary_rows = []
    mist_labels = []
    evidence = []
    root_only_ids = []
    for row in labels:
        spectrum_id = row["spec"]
        formula = row["formula"]
        ion = row["ionization"]
        root_counts = _official_mist_formula_counts(formula)
        if root_counts is None:
            raise ValueError(f"Official MIST cannot parse root formula {formula!r}")
        ppm_error = float(row["precursor_ppm_error"])
        if not math.isfinite(ppm_error) or ppm_error > PRECURSOR_PPM_TOLERANCE:
            raise ValueError(
                f"Invalid precursor ppm error for {spectrum_id}: {ppm_error}"
            )

        subformula_path = subformula_dir / f"{spectrum_id}.json"
        payload = json.loads(subformula_path.read_text())
        selected = payload.get(formula)
        if selected is None:
            raise ValueError(
                f"MIST-CF subformula file lacks top-1 {formula} for {spectrum_id}"
            )
        if selected.get("cand_ion") != ion:
            raise ValueError(
                f"MIST-CF subformula ion mismatch for {spectrum_id}: "
                f"expected={ion} observed={selected.get('cand_ion')}"
            )

        fragments = [
            {
                "id": 0,
                "molecularFormula": formula,
                "relativeIntensity": 0.0,
                "mz": float(row["parentmass"]),
            }
        ]
        table = selected.get("cand_tbl")
        if table is None or not table.get("formula"):
            root_only_ids.append(spectrum_id)
        else:
            required = ("formula", "mz", "ms2_inten", "mass_diff", "ions")
            if any(key not in table for key in required):
                raise ValueError(
                    f"Incomplete MIST-CF subformula table for {spectrum_id}"
                )
            lengths = {len(table[key]) for key in required}
            if len(lengths) != 1:
                raise ValueError(
                    f"Misaligned MIST-CF subformula table for {spectrum_id}"
                )
            for index, (
                fragment_formula,
                mz,
                intensity,
                assignment_ppm,
                fragment_ion,
            ) in enumerate(
                zip(
                    table["formula"],
                    table["mz"],
                    table["ms2_inten"],
                    table["mass_diff"],
                    table["ions"],
                ),
                start=1,
            ):
                _validate_fragment_formula(fragment_formula, root_counts)
                mz = float(mz)
                intensity = float(intensity)
                assignment_ppm = float(assignment_ppm)
                if not math.isfinite(mz) or mz <= 0:
                    raise ValueError(
                        f"Invalid MIST-CF fragment m/z for {spectrum_id}: {mz}"
                    )
                if not math.isfinite(intensity) or intensity < 0:
                    raise ValueError(
                        f"Invalid MIST-CF fragment intensity for {spectrum_id}: {intensity}"
                    )
                if not math.isfinite(assignment_ppm) or assignment_ppm < 0:
                    raise ValueError(
                        f"Invalid MIST-CF assignment ppm for {spectrum_id}: "
                        f"{assignment_ppm}"
                    )
                if assignment_ppm > SUBFORMULA_PPM_TOLERANCE:
                    raise ValueError(
                        f"MIST-CF assignment exceeds {SUBFORMULA_PPM_TOLERANCE} ppm "
                        f"for {spectrum_id}: {assignment_ppm}"
                    )
                if fragment_ion != ion:
                    raise ValueError(
                        f"MIST-CF fragment ion mismatch for {spectrum_id}: "
                        f"expected={ion} observed={fragment_ion}"
                    )
                fragments.append(
                    {
                        "id": index,
                        "molecularFormula": fragment_formula,
                        "relativeIntensity": intensity,
                        "mz": mz,
                    }
                )

        tree = {
            "molecularFormula": formula,
            "annotations": {
                "precursorIonType": ion,
                "source": "official MIST-CF top-1 peak-to-subformula assignment",
            },
            "fragments": fragments,
            "losses": [],
        }
        tree_path = trees_dir / f"{spectrum_id}.json"
        tree_path.write_text(json.dumps(tree, indent=2, sort_keys=True) + "\n")
        spec_path = output_dir / "spec_files" / f"{spectrum_id}.ms"
        summary_rows.append(
            {
                "": len(summary_rows),
                "spec_name": spectrum_id,
                "spec_file": str(spec_path.resolve()),
                "tree_file": str(tree_path.resolve()),
            }
        )
        mist_labels.append(
            {
                "dataset": output_dir.name,
                "spec": spectrum_id,
                "formula": formula,
                "ionization": ion,
                "parentmass": row["parentmass"],
            }
        )
        evidence.append(
            {
                "spec": spectrum_id,
                "formula": formula,
                "ionization": ion,
                "precursor_ppm_error": row["precursor_ppm_error"],
                "subformula_file_sha256": sha256_file(subformula_path),
                "selected_subformula_sha256": _json_digest(selected),
                "peakformula_tree_sha256": sha256_file(tree_path),
                "fragment_rows": len(fragments) - 1,
                "maximum_assignment_ppm": max(
                    (float(value) for value in table["mass_diff"]), default=None
                )
                if table is not None
                else None,
            }
        )

    summary_path = summary_dir / "summary_df.tsv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["", "spec_name", "spec_file", "tree_file"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    mist_labels_path = output_dir / "mist_labels.tsv"
    with mist_labels_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "spec", "formula", "ionization", "parentmass"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(mist_labels)

    evidence_path = output_dir / "feature_bridge_rows.jsonl"
    with evidence_path.open("w") as handle:
        for row in evidence:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    manifest = {
        "schema_version": 1,
        "kind": FEATURE_BRIDGE_KIND,
        "formula_source": FORMULA_SOURCE,
        "rows": len(evidence),
        "top1_candidate_rows": len(evidence),
        "root_only_rows": len(root_only_ids),
        "root_only_ids": root_only_ids,
        "maximum_precursor_ppm_error": max(
            float(row["precursor_ppm_error"]) for row in labels
        ),
        "maximum_subformula_assignment_ppm": max(
            (
                row["maximum_assignment_ppm"]
                for row in evidence
                if row["maximum_assignment_ppm"] is not None
            ),
            default=None,
        ),
        "subformula_assignment_ppm_tolerance": SUBFORMULA_PPM_TOLERANCE,
        "formula_manifest_sha256": sha256_file(
            output_dir / "formula_bridge_manifest.json"
        ),
        "mist_cf_subformula_dir": str(subformula_dir.resolve()),
        "mist_labels_sha256": sha256_file(mist_labels_path),
        "summary_sha256": sha256_file(summary_path),
        "per_id_mapping_sha256": sha256_file(evidence_path),
        "mist_cf_subformula_evidence_sha256": _json_digest(
            [
                {
                    "spec": row["spec"],
                    "subformula_file_sha256": row["subformula_file_sha256"],
                    "selected_subformula_sha256": row["selected_subformula_sha256"],
                }
                for row in evidence
            ]
        ),
        "peakformula_tree_evidence_sha256": _json_digest(
            [
                {
                    "spec": row["spec"],
                    "peakformula_tree_sha256": row["peakformula_tree_sha256"],
                }
                for row in evidence
            ]
        ),
        "mist_cf_git_commit": mist_cf_git_commit,
        "mist_cf_checkpoint_sha256": mist_cf_checkpoint_sha256,
        "compatibility_layout_note": (
            "Official MIST hard-codes sirius_outputs/summary_statistics/summary_df.tsv; "
            "the trees here come directly from official MIST-CF, not SIRIUS"
        ),
    }
    manifest_path = output_dir / "feature_bridge_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mgf", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--subformula-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mist-cf-git-commit", required=True)
    parser.add_argument("--mist-cf-checkpoint-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            build_peakformula_dataset(
                args.mgf,
                args.predictions,
                args.subformula_dir,
                args.output_dir,
                args.mist_cf_git_commit,
                args.mist_cf_checkpoint_sha256,
            )
        )
    )


if __name__ == "__main__":
    main()
