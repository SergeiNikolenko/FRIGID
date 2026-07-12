from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from frigid.rankloop_corpus import (
    build_rankloop_corpus,
    load_candidate_source,
    load_spectrum_records,
    molecule_record_from_smiles,
    split_partition_groups,
    write_rankloop_corpus,
)


def _write_dataset(tmp_path: Path) -> tuple[Path, Path]:
    labels = pd.DataFrame(
        [
            ("q1a", "[M+H]+", "C2H6O", "CCO", "LFQSCWFLJHTTHZ", "Orbitrap"),
            ("q1b", "[M+H]+", "C2H6O", "CCO", "LFQSCWFLJHTTHZ", "Orbitrap"),
            ("q2", "[M+H]+", "C2H6O", "COC", "LCGLNKUTAGEVQW", "Orbitrap"),
            ("q3", "[M+H]+", "C3H8O", "CCCO", "BDERNNFJNOPAEC", "QTOF"),
            ("q4", "[M+H]+", "C3H8O", "CC(C)O", "KFZMGEQAYNKOFK", "QTOF"),
            ("q5", "[M+H]+", "C7H8", "Cc1ccccc1", "YXFVVABEGXRONW", "Orbitrap"),
            ("q6", "[M+H]+", "C8H10", "CCc1ccccc1", "YNQLUTRBYVCPMQ", "Orbitrap"),
            ("test_only", "[M+H]+", "CH4O", "CO", "OKKJLVBELUTLKV", "Orbitrap"),
        ],
        columns=("spec", "ionization", "formula", "smiles", "inchikey", "instrument"),
    )
    labels.insert(0, "dataset", "MassSpecGym")
    splits = pd.DataFrame(
        [(name, "test" if name == "test_only" else "train") for name in labels["spec"]],
        columns=("name", "split"),
    )
    labels_path = tmp_path / "labels.tsv"
    split_path = tmp_path / "split.tsv"
    labels.to_csv(labels_path, sep="\t", index=False)
    splits.to_csv(split_path, sep="\t", index=False)
    return labels_path, split_path


def test_scaffold_groups_are_partitioned_together(tmp_path: Path):
    labels, splits = _write_dataset(tmp_path)
    records, _ = load_spectrum_records(labels, splits, fingerprint_bits=128)
    partitions = split_partition_groups(records, development_fraction=0.4, seed=7)

    aromatic = [record for record in records if record.spec_name in {"q5", "q6"}]
    assert aromatic[0].molecule.scaffold == aromatic[1].molecule.scaffold
    assert (
        partitions[aromatic[0].molecule.partition_group]
        == partitions[aromatic[1].molecule.partition_group]
    )


def test_candidate_source_rejects_supervision_columns(tmp_path: Path):
    source = tmp_path / "source.csv"
    pd.DataFrame(
        [{"query_spec_name": "q1", "candidate_smiles": "CCO", "target_smiles": "CCO"}]
    ).to_csv(source, index=False)

    with pytest.raises(ValueError, match="forbidden supervision"):
        load_candidate_source(
            "unsafe",
            source,
            allowed_queries={"q1"},
            fingerprint_bits=128,
            fingerprint_radius=2,
        )


def test_computed_connectivity_is_used_when_label_inchikey_disagrees():
    molecule = molecule_record_from_smiles(
        "CCO",
        inchi_key="AAAAAAAAAAAAAA",
        fingerprint_bits=128,
    )

    assert molecule is not None
    assert molecule.inchi_key_first_block == "LFQSCWFLJHTTHZ"
    assert molecule.provided_inchi_key_first_block == "AAAAAAAAAAAAAA"


def test_corpus_is_leakage_safe_and_deterministic(tmp_path: Path):
    labels, splits = _write_dataset(tmp_path)
    records, rejections = load_spectrum_records(labels, splits, fingerprint_bits=128)
    assert not rejections
    assert {record.spec_name for record in records}.isdisjoint({"test_only"})

    first, first_report = build_rankloop_corpus(
        records,
        development_fraction=0.4,
        negatives_per_query=2,
        max_spectra_per_molecule=2,
        seed=11,
    )
    second, second_report = build_rankloop_corpus(
        records,
        development_fraction=0.4,
        negatives_per_query=2,
        max_spectra_per_molecule=2,
        seed=11,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first_report == second_report
    assert first.groupby("query_spec_name")["label"].sum().eq(1).all()
    assert first.groupby("query_spec_name").size().eq(3).all()
    assert "OKKJLVBELUTLKV" not in set(first["candidate_inchi_key_first_block"])
    assert first_report["connectivity_overlap_count"] == 0
    assert first_report["scaffold_overlap_count"] == 0

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    common = {
        "labels_tsv": labels,
        "split_tsv": splits,
        "source_paths": [],
        "parameters": {"seed": 11},
        "repo_root": Path(__file__).parents[1],
    }
    first_manifest = write_rankloop_corpus(
        first,
        first_report,
        output_dir=first_dir,
        **common,
    )
    second_manifest = write_rankloop_corpus(
        second,
        second_report,
        output_dir=second_dir,
        **common,
    )
    assert (
        first_manifest["outputs"]["candidate_corpus_sha256"]
        == second_manifest["outputs"]["candidate_corpus_sha256"]
    )
