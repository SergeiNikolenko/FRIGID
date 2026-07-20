import csv
import zipfile

import pytest

from scripts.audit_mist_cf_split import audit_archive_connectivity, audit_split


def write_rows(path, fieldnames, rows, delimiter):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def test_audit_mist_cf_split_reports_training_overlap(tmp_path):
    split = tmp_path / "split.tsv"
    metadata = tmp_path / "metadata.csv"
    write_rows(
        split,
        ["spec", "Fold_0"],
        [
            {"spec": "a", "Fold_0": "train"},
            {"spec": "b", "Fold_0": "test"},
        ],
        "\t",
    )
    write_rows(
        metadata,
        ["spec_name"],
        [{"spec_name": "a"}, {"spec_name": "b"}],
        ",",
    )

    result = audit_split(split, metadata)

    assert result["evaluation_split_counts"] == {"train": 1, "test": 1}
    assert result["train_or_validation_overlap"] == 1
    assert result["released_checkpoint_is_evaluation_disjoint"] is False


def test_audit_mist_cf_split_rejects_missing_evaluation_id(tmp_path):
    split = tmp_path / "split.tsv"
    metadata = tmp_path / "metadata.csv"
    write_rows(
        split,
        ["spec", "Fold_0"],
        [{"spec": "a", "Fold_0": "test"}],
        "\t",
    )
    write_rows(
        metadata,
        ["spec_name"],
        [{"spec_name": "missing"}],
        ",",
    )

    with pytest.raises(ValueError, match="missing 1 evaluation IDs"):
        audit_split(split, metadata)


def test_archive_connectivity_audit_finds_duplicate_structure(tmp_path):
    split = tmp_path / "split.tsv"
    metadata = tmp_path / "metadata.csv"
    archive = tmp_path / "train.zip"
    write_rows(
        split,
        ["spec", "Fold_0"],
        [
            {"spec": "evaluation", "Fold_0": "test"},
            {"spec": "duplicate", "Fold_0": "train"},
            {"spec": "other", "Fold_0": "train"},
        ],
        "\t",
    )
    write_rows(
        metadata,
        ["spec_name", "inchikey_first_block"],
        [{"spec_name": "evaluation", "inchikey_first_block": "ABCDEFGHIJKLMN"}],
        ",",
    )
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(
            "canopus_train/spec_files/evaluation.ms",
            ">InChIKey ABCDEFGHIJKLMN-UHFFFAOYSA-N\n>ms2peaks\n",
        )
        handle.writestr(
            "canopus_train/spec_files/duplicate.ms",
            "#InChIKey ABCDEFGHIJKLMN-OTHERBLOCK-X\n>ms2peaks\n",
        )
        handle.writestr(
            "canopus_train/spec_files/other.ms",
            ">InChIKey ZZZZZZZZZZZZZZ-UHFFFAOYSA-N\n>ms2peaks\n",
        )

    result = audit_archive_connectivity(archive, split, metadata)

    assert result["overlap_split_counts"] == {"test": 1, "train": 1}
    assert result["overlap_unique_connectivities"] == 1
    assert result["train_or_validation_connectivity_overlap"] == 1
    assert result["released_checkpoint_is_connectivity_disjoint"] is False
