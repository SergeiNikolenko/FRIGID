import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "export_mist_fingerprints.py"
SPEC = importlib.util.spec_from_file_location("export_mist_fingerprints", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeSpectrum:
    def __init__(self, name: str):
        self.name = name

    def get_spec_name(self) -> str:
        return self.name


def split_data(*names: str):
    return [(FakeSpectrum(name), object()) for name in names]


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.values = [10, 20, 30]

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, item):
        return self.values[item]

    def get_featurizer(self):
        return "mock-featurizer"


def test_resolve_export_indices_preserves_manifest_order(tmp_path: Path):
    manifest = tmp_path / "subset.csv"
    manifest.write_text("spec_name\nC\nA\n")

    indices = MODULE.resolve_export_indices(
        split_data("A", "B", "C"),
        str(manifest),
        start_index=0,
        max_spectra=None,
    )

    assert indices == [2, 0]


def test_build_export_subset_uses_selected_order_and_preserves_featurizer():
    selected_dataset, selected_indices = MODULE.build_export_subset(
        FakeDataset(),
        split_data("A", "B", "C"),
        None,
        start_index=1,
        max_spectra=2,
    )

    assert selected_indices == [1, 2]
    assert len(selected_dataset) == 2
    assert selected_dataset.get_featurizer() == "mock-featurizer"
    assert selected_dataset[0] == 20
    assert selected_dataset[1] == 30


def test_resolve_export_indices_rejects_manifest_start_and_truncation(tmp_path: Path):
    manifest = tmp_path / "subset.tsv"
    manifest.write_text("spec_name\nC\n")

    with pytest.raises(ValueError, match="start_index"):
        MODULE.resolve_export_indices(
            split_data("A", "B", "C"),
            str(manifest),
            start_index=1,
            max_spectra=None,
        )

    with pytest.raises(ValueError, match="cannot truncate"):
        MODULE.resolve_export_indices(
            split_data("A", "B", "C"),
            str(manifest),
            start_index=0,
            max_spectra=2,
        )
