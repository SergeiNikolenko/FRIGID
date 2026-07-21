import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.package_mist_predictions import package_predictions
from scripts.prepare_mist_predicted_formula_dataset import (
    FORMULA_SOURCE,
    build_dataset,
    select_mass_consistent_candidate,
)
from scripts.audit_sirius_formula_bridge import audit_project
from scripts.unpack_sirius_for_mist import (
    unpack_project,
    write_bridge_manifest,
    write_mist_labels,
    write_summary,
)


def test_sirius_job_uses_import_safe_naming_convention() -> None:
    job_script = (
        Path(__file__).parents[1]
        / "scripts/slurm_mist_predicted_formula_fingerprints.sbatch"
    ).read_text()

    assert "--naming-convention '%compoundname'" in job_script
    assert "--naming-convention '%index_%compoundname'" not in job_script
    assert "#SBATCH --gres=" not in job_script
    assert "  --gpu \\" not in job_script
    assert 'export PYTHONPATH="$MIST/src"' in job_script
    assert 'from mist import pred_fp' in job_script


def test_formula_bridge_selects_top_prediction_and_preserves_order(tmp_path: Path) -> None:
    mgf = tmp_path / "test.mgf"
    mgf.write_text(
        "BEGIN IONS\nSCANS=a\nPEPMASS=43.05422664\n10 1\nEND IONS\n"
        "BEGIN IONS\nFEATURE_ID=b\nPEPMASS=57.06987670\n20 2\nEND IONS\n"
    )
    predictions = tmp_path / "predictions.tsv"
    predictions.write_text(
        "spec\tcand_form\tscores\tcand_ion\tparentmasses\n"
        "a\tC2H4\t0.1\t[M+H]+\t29.03857658\n"
        "a\tC3H6\t0.9\t[M+H]+\t43.05422664\n"
        "b\tC4H8\t0.5\t[M+H]+\t57.06987670\n"
    )

    manifest = build_dataset(mgf, predictions, tmp_path / "dataset")

    labels = pd.read_csv(tmp_path / "dataset/labels.tsv", sep="\t")
    assert labels["spec"].tolist() == ["a", "b"]
    assert labels["formula"].tolist() == ["C3H6", "C4H8"]
    assert manifest["formula_source"] == FORMULA_SOURCE
    assert labels["candidate_rank"].tolist() == [1, 1]
    assert "FORMULA=C3H6" in (tmp_path / "dataset/forced_formula.mgf").read_text()
    assert ">formula C4H8" in (tmp_path / "dataset/spec_files/b.ms").read_text()


def test_formula_bridge_advances_to_requested_candidate_rank() -> None:
    candidates = [
        {
            "cand_form": "C3H6",
            "cand_ion": "[M+H]+",
            "scores": "0.9",
            "parentmasses": "43.05422664",
        },
        {
            "cand_form": "C3H6",
            "cand_ion": "[M+H]+",
            "scores": "0.8",
            "parentmasses": "43.05422664",
        },
    ]

    _, rank, _, _ = select_mass_consistent_candidate(
        candidates, 43.05422664, minimum_rank=2
    )

    assert rank == 2


def _write_zip(path: Path, member: str, content: str) -> None:
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr(member, content)


def test_sirius_unpack_and_mist_packaging_are_id_locked(tmp_path: Path) -> None:
    labels = tmp_path / "labels.tsv"
    labels.write_text("dataset\tspec\tformula\tionization\tparentmass\nset\ta\tC2H4\t[M+H]+\t29\n")
    compound = tmp_path / "project/0_a"
    compound.mkdir(parents=True)
    tree = {
        "molecularFormula": "C2H4",
        "annotations": {"precursorIonType": "[M + H]+"},
        "fragments": [],
        "losses": [],
    }
    _write_zip(compound / "trees", "C2H4_[M+H]+.json", json.dumps(tree))
    _write_zip(compound / "spectra", "C2H4_[M+H]+.tsv", "mz\tintensity\n")
    _write_zip(compound / "scores", "C2H4_[M+H]+.info", "score\n")
    (compound / "compound.info").write_text(
        "name\ta\nionMass\t29.03857658\nionType\t[M + H]+\n"
    )
    (compound / "spectrum.ms").write_text(
        ">compound a\n>formula C2H4\n>ionization [M + H]+\n"
    )

    rows = unpack_project(tmp_path / "project", labels)
    summary = tmp_path / "project/summary_statistics/summary_df.tsv"
    write_summary(rows, summary)
    mist_labels = tmp_path / "mist_labels.tsv"
    write_mist_labels(rows, mist_labels)

    assert rows[0]["spec_name"] == "a"
    assert rows[0]["pred_formula"] == "C2H4"
    assert rows[0]["mist_cf_formula"] == "C2H4"
    assert rows[0]["tree_formula"] == "C2H4"
    assert rows[0]["formula_normalization"] == "identity"
    assert Path(rows[0]["tree_file"]).is_file()
    summary_frame = pd.read_csv(summary, sep="\t", index_col=0)
    assert summary_frame["spec_name"].tolist() == ["a"]
    assert pd.read_csv(mist_labels, sep="\t")["formula"].tolist() == ["C2H4"]

    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {"fingerprint_index": [0, 1], "spec_name": ["a", "b"]}
    ).to_csv(metadata, index=False)
    predictions = tmp_path / "mist.p"
    with predictions.open("wb") as handle:
        pickle.dump(
            {
                "dataset_name": mist_labels.parent.name,
                "args": {"labels_name": mist_labels.name},
                "names": ["b", "a"],
                "preds": np.stack(
                    [np.full(4096, 2.0), np.full(4096, 1.0)]
                ),
            },
            handle,
        )
    formula_manifest = tmp_path / "formula_manifest.json"
    formula_manifest.write_text(
        json.dumps(
            {
                    "kind": "Mass-consistent MIST-CF predicted-formula bridge into official MIST",
                    "formula_source": FORMULA_SOURCE,
                    "precursor_ppm_tolerance": 10.0,
                    "fallback_rows": 1,
                    "maximum_candidate_rank": 2,
                    "rows": 2,
                    "sirius_consistency_validated": True,
            }
        )
    )
    from scripts.package_mist_predictions import sha256_file

    sirius_bridge_manifest = tmp_path / "sirius_bridge_manifest.json"
    sirius_bridge_manifest.write_text(
        json.dumps(
            {
                "kind": "SIRIUS-validated formula-blind bridge into official MIST",
                "formula_source": FORMULA_SOURCE,
                "rows": 2,
                "formula_manifest_sha256": sha256_file(formula_manifest),
                "mist_labels_sha256": sha256_file(mist_labels),
                "maximum_mist_precursor_ppm_error": 2.0,
                "sirius_audit_sha256": "audit-sha256",
                "summary_sha256": "summary-sha256",
                "sirius_tree_evidence_sha256": "tree-evidence-sha256",
                "per_id_mapping_sha256": "mapping-sha256",
            }
        )
    )
    output = tmp_path / "fingerprints.npz"

    manifest = package_predictions(
        predictions,
        metadata,
        formula_manifest,
        sirius_bridge_manifest,
        mist_labels,
        output,
        "mist-commit",
        "5.5.7",
        "checkpoint-sha256",
    )

    with np.load(output) as bundle:
        assert bundle["spectrum_ids"].tolist() == ["a", "b"]
        assert bundle["probs"][:, 0].tolist() == [1.0, 2.0]
    assert manifest["rows"] == 2
    assert manifest["official_mist_git_commit"] == "mist-commit"
    assert manifest["fallback_rows"] == 1


def test_sirius_unpack_rejects_tree_not_matching_mist_cf_formula(tmp_path: Path) -> None:
    labels = tmp_path / "labels.tsv"
    labels.write_text(
        "dataset\tspec\tformula\tionization\tparentmass\n"
        "set\ta\tC2H4\t[M+H]+\t29\n"
    )
    compound = tmp_path / "project/0_a"
    compound.mkdir(parents=True)
    tree = {
        "molecularFormula": "C3H6",
        "annotations": {"precursorIonType": "[M + H]+"},
    }
    _write_zip(compound / "trees", "tree.json", json.dumps(tree))
    _write_zip(compound / "spectra", "spectrum.tsv", "mz\tintensity\n")
    _write_zip(compound / "scores", "score.info", "score\n")
    (compound / "compound.info").write_text(
        "name\ta\nionMass\t29.0\nionType\t[M + H]+\n"
    )
    (compound / "spectrum.ms").write_text(
        ">compound a\n>formula C2H4\n>ionization [M + H]+\n"
    )

    try:
        unpack_project(tmp_path / "project", labels)
    except ValueError as error:
        assert "SIRIUS formula mismatch" in str(error)
    else:
        raise AssertionError("mismatched SIRIUS formula was accepted")


def test_sirius_unpack_accepts_documented_radical_cation_normalization(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "labels.tsv"
    labels.write_text(
        "dataset\tspec\tformula\tionization\tparentmass\n"
        "set\ta\tC10H12N4O2\t[M]+\t220.095\n"
    )
    compound = tmp_path / "project/190_a"
    compound.mkdir(parents=True)
    tree = {
        "molecularFormula": "C10H11N4O2",
        "annotations": {"precursorIonType": "[M]+"},
    }
    _write_zip(compound / "trees", "tree.json", json.dumps(tree))
    _write_zip(compound / "spectra", "spectrum.tsv", "mz\tintensity\n")
    _write_zip(compound / "scores", "score.info", "score\n")
    (compound / "compound.info").write_text(
        "name\ta\nionMass\t220.095\nionType\t[M]+\n"
    )
    (compound / "spectrum.ms").write_text(
        ">compound a\n>formula C10H12N4O2\n>ionization [M]+\n"
    )

    rows = unpack_project(tmp_path / "project", labels)

    assert rows[0]["pred_formula"] == "C10H11N4O2"
    assert rows[0]["mist_cf_formula"] == "C10H12N4O2"
    assert rows[0]["tree_formula"] == "C10H11N4O2"
    assert rows[0]["formula_normalization"] == "sirius_[M]+_minus_H"
    assert rows[0]["mist_adduct"] == "[M+H]+"
    assert float(rows[0]["mist_precursor_ppm_error"]) < 10.0


def test_sirius_audit_requests_next_rank_for_changed_adduct(tmp_path: Path) -> None:
    labels = tmp_path / "labels.tsv"
    labels.write_text(
        "dataset\tspec\tformula\tionization\tparentmass\tcandidate_rank\n"
        "set\ta\tC2H4\t[M+H]+\t29.03857658\t1\n"
    )
    compound = tmp_path / "project/0_a"
    compound.mkdir(parents=True)
    tree = {
        "molecularFormula": "C2H4",
        "annotations": {"precursorIonType": "[M+Na]+"},
    }
    _write_zip(compound / "trees", "tree.json", json.dumps(tree))
    (compound / "compound.info").write_text(
        "name\ta\nionMass\t29.03857658\nionType\t[M+Na]+\n"
    )
    (compound / "spectrum.ms").write_text(
        ">compound a\n>formula C2H4\n>ionization [M+H]+\n"
    )

    report = audit_project(
        tmp_path / "project", labels, tmp_path / "audit.json"
    )

    assert report["mismatch_count"] == 1
    assert report["minimum_candidate_ranks"] == {"a": 2}


def test_sirius_audit_rejects_signed_formula_unsupported_by_official_mist(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "labels.tsv"
    labels.write_text(
        "dataset\tspec\tformula\tionization\tparentmass\tcandidate_rank\n"
        "set\ta\tC8H12N5\t[M-H2O+H]+\t161.106\t1\n"
    )
    compound = tmp_path / "project/0_a"
    compound.mkdir(parents=True)
    (tmp_path / "project/summary_statistics").mkdir()
    tree = {
        "molecularFormula": "C8H12N5",
        "annotations": {"precursorIonType": "[M-H2O+H]+"},
        "fragments": [
            {"id": 0, "molecularFormula": "C8H12N5"},
            {"id": 1, "molecularFormula": "C8H10N5-O"},
        ],
        "losses": [
            {"source": 0, "target": 1, "molecularFormula": "H2O"}
        ],
    }
    _write_zip(compound / "trees", "tree.json", json.dumps(tree))
    (compound / "compound.info").write_text(
        "name\ta\nionMass\t161.106\nionType\t[M-H2O+H]+\n"
    )
    (compound / "spectrum.ms").write_text(
        ">compound a\n>formula C8H12N5\n>ionization [M-H2O+H]+\n"
    )

    report = audit_project(tmp_path / "project", labels, tmp_path / "audit.json")

    assert report["mismatch_count"] == 1
    mismatch = report["mismatches"][0]
    assert mismatch["reasons"] == ["official_mist_tree_incompatible"]
    assert mismatch["official_mist_tree_issues"] == [
        {"kind": "fragment_formula", "fragment_id": 1, "formula": "C8H10N5-O"}
    ]
    assert report["minimum_candidate_ranks"] == {"a": 2}


def test_bridge_manifest_rejects_audit_not_bound_to_formula_manifest(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "labels.tsv"
    labels.write_text("spec\n")
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "mismatch_count": 0,
                "rows": 0,
                "labels_sha256": "wrong",
            }
        )
    )
    formula_manifest = tmp_path / "formula.json"
    formula_manifest.write_text(
        json.dumps(
            {
                "sirius_consistency_validated": True,
                "sirius_consistency_audit_sha256": "wrong",
            }
        )
    )

    with pytest.raises(ValueError, match="not bound to the supplied SIRIUS audit"):
        write_bridge_manifest(
            [],
            formula_manifest,
            audit,
            labels,
            tmp_path / "summary.tsv",
            tmp_path / "mist_labels.tsv",
            tmp_path / "bridge.json",
        )
