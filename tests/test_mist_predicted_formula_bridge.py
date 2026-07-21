import json
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.package_mist_predictions import package_predictions
from scripts.prepare_mist_predicted_formula_dataset import build_dataset
from scripts.unpack_sirius_for_mist import unpack_project, write_summary


def test_sirius_job_uses_import_safe_naming_convention() -> None:
    job_script = (
        Path(__file__).parents[1]
        / "scripts/slurm_mist_predicted_formula_fingerprints.sbatch"
    ).read_text()

    assert "--naming-convention '%compoundname'" in job_script
    assert "--naming-convention '%index_%compoundname'" not in job_script
    assert "#SBATCH --gres=" not in job_script
    assert "  --gpu \\" not in job_script


def test_formula_bridge_selects_top_prediction_and_preserves_order(tmp_path: Path) -> None:
    mgf = tmp_path / "test.mgf"
    mgf.write_text(
        "BEGIN IONS\nSCANS=a\nPEPMASS=101.0\n10 1\nEND IONS\n"
        "BEGIN IONS\nFEATURE_ID=b\nPEPMASS=202.0\n20 2\nEND IONS\n"
    )
    predictions = tmp_path / "predictions.tsv"
    predictions.write_text(
        "spec\tcand_form\tscores\tcand_ion\tparentmasses\n"
        "a\tC2H4\t0.1\t[M+H]+\t101.0\n"
        "a\tC3H6\t0.9\t[M+H]+\t101.0\n"
        "b\tC4H8\t0.5\t[M+H]+\t202.0\n"
    )

    manifest = build_dataset(mgf, predictions, tmp_path / "dataset")

    labels = pd.read_csv(tmp_path / "dataset/labels.tsv", sep="\t")
    assert labels["spec"].tolist() == ["a", "b"]
    assert labels["formula"].tolist() == ["C3H6", "C4H8"]
    assert manifest["formula_source"] == "MIST-CF top-1 prediction; no ground-truth formula"
    assert "FORMULA=C3H6" in (tmp_path / "dataset/forced_formula.mgf").read_text()
    assert ">formula C4H8" in (tmp_path / "dataset/spec_files/b.ms").read_text()


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
    (compound / "compound.info").write_text("ionMass\t29.0\n")

    rows = unpack_project(tmp_path / "project", labels)
    summary = tmp_path / "project/summary_statistics/summary_df.tsv"
    write_summary(rows, summary)

    assert rows[0]["spec_name"] == "a"
    assert rows[0]["pred_formula"] == "C2H4"
    assert Path(rows[0]["tree_file"]).is_file()
    summary_frame = pd.read_csv(summary, sep="\t", index_col=0)
    assert summary_frame["spec_name"].tolist() == ["a"]

    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {"fingerprint_index": [0, 1], "spec_name": ["a", "b"]}
    ).to_csv(metadata, index=False)
    predictions = tmp_path / "mist.p"
    with predictions.open("wb") as handle:
        pickle.dump(
            {
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
                "kind": "MIST-CF top-1 predicted-formula bridge into official MIST",
                "formula_source": "MIST-CF top-1 prediction; no ground-truth formula",
                "rows": 2,
            }
        )
    )
    output = tmp_path / "fingerprints.npz"

    manifest = package_predictions(
        predictions,
        metadata,
        formula_manifest,
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
    (compound / "compound.info").write_text("ionMass\t29.0\n")

    try:
        unpack_project(tmp_path / "project", labels)
    except ValueError as error:
        assert "SIRIUS formula mismatch" in str(error)
    else:
        raise AssertionError("mismatched SIRIUS formula was accepted")
