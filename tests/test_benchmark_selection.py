from pathlib import Path

import pytest

from dlm.utils.benchmark_selection import (
    hash_spec_names,
    load_spec_manifest,
    resolve_selected_indices,
)


class FakeSpectrum:
    def __init__(self, name: str):
        self.name = name

    def get_spec_name(self) -> str:
        return self.name


def split_data(*names: str):
    return [(FakeSpectrum(name), object()) for name in names]


def test_manifest_preserves_order_and_hashes_names(tmp_path: Path):
    manifest = tmp_path / "subset.tsv"
    manifest.write_text("spec_name\nB\nA\n")

    names = load_spec_manifest(manifest)

    assert names == ["B", "A"]
    assert hash_spec_names(names) == hash_spec_names(["B", "A"])
    assert hash_spec_names(names) != hash_spec_names(["A", "B"])


def test_manifest_rejects_duplicates(tmp_path: Path):
    manifest = tmp_path / "subset.csv"
    manifest.write_text("spec_name\nA\nA\n")

    with pytest.raises(ValueError, match="duplicate"):
        load_spec_manifest(manifest)


def test_resolve_selected_indices_uses_manifest_order():
    indices = resolve_selected_indices(
        split_data("A", "B", "C"),
        ["C", "A"],
        start_index=0,
        max_spectra=2,
    )

    assert indices == [2, 0]


def test_resolve_selected_indices_rejects_slice_with_manifest():
    with pytest.raises(ValueError, match="start_index"):
        resolve_selected_indices(
            split_data("A", "B"),
            ["A"],
            start_index=1,
            max_spectra=None,
        )
