from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "export_diffms_mist_fingerprints.py"
SPEC = importlib.util.spec_from_file_location("export_diffms_mist_fingerprints", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_locked_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    metadata_path = tmp_path / "metadata.csv"
    labels_path = tmp_path / "labels.tsv"
    split_path = tmp_path / "split.tsv"
    spectra_dir = tmp_path / "spectra"
    subformula_dir = tmp_path / "subformulae"
    spectra_dir.mkdir()
    subformula_dir.mkdir()
    pd.DataFrame(
        {
            "fingerprint_index": [1, 0],
            "spec_name": ["spec-b", "spec-a"],
        }
    ).to_csv(metadata_path, index=False)
    pd.DataFrame(
        {
            "spec": ["spec-a", "spec-b"],
            "formula": ["C2H6O", "C3H8O"],
            "instrument": ["Orbitrap", "QTOF"],
        }
    ).to_csv(labels_path, sep="\t", index=False)
    pd.DataFrame(
        {"name": ["spec-a", "spec-b"], "split": ["val", "val"]}
    ).to_csv(split_path, sep="\t", index=False)
    for spectrum_id in ("spec-a", "spec-b"):
        (spectra_dir / f"{spectrum_id}.ms").write_text("#INSTRUMENT TYPE test\n")
        (subformula_dir / f"{spectrum_id}.json").write_text(
            '{"cand_form":"C2H6O","cand_ion":"[M+H]+","output_tbl":null}\n'
        )
    return metadata_path, labels_path, split_path, spectra_dir, subformula_dir


def test_load_locked_rows_restores_reference_order_and_hashes_inputs(tmp_path):
    metadata, labels, split, spectra, subformulae = _write_locked_inputs(tmp_path)

    rows, subformula_hash = MODULE.load_locked_rows(
        metadata,
        labels,
        split,
        spectra,
        subformulae,
        id_column="spec_name",
        index_column="fingerprint_index",
        expected_split="val",
        expected_rows=2,
    )

    assert [row.spectrum_id for row in rows] == ["spec-a", "spec-b"]
    assert [row.formula for row in rows] == ["C2H6O", "C3H8O"]
    assert len(subformula_hash) == 64


def test_load_locked_rows_rejects_wrong_split(tmp_path):
    metadata, labels, split, spectra, subformulae = _write_locked_inputs(tmp_path)
    split_frame = pd.read_csv(split, sep="\t")
    split_frame.loc[0, "split"] = "train"
    split_frame.to_csv(split, sep="\t", index=False)

    with pytest.raises(ValueError, match="outside expected split"):
        MODULE.load_locked_rows(
            metadata,
            labels,
            split,
            spectra,
            subformulae,
            id_column="spec_name",
            index_column="fingerprint_index",
            expected_split="val",
            expected_rows=2,
        )


def test_extract_encoder_state_removes_only_the_exact_prefix():
    checkpoint = {
        "state_dict": {
            "encoder.weight": torch.ones(2, 2),
            "decoder.weight": torch.zeros(2, 2),
        }
    }

    state = MODULE.extract_encoder_state(checkpoint)

    assert list(state) == ["weight"]
    assert torch.equal(state["weight"], torch.ones(2, 2))


class _TinyDataset(Dataset):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int):
        return {"features": torch.tensor([float(index)]), "name": f"spec-{index}"}


def _collate(rows):
    return {
        "features": torch.stack([row["features"] for row in rows]),
        "names": [row["name"] for row in rows],
    }


class _TinyEncoder(nn.Module):
    def forward(self, batch):
        values = batch["features"]
        logits = torch.cat([values, -values, values + 1.0, values - 1.0], dim=1)
        return torch.sigmoid(logits), {"unused": values}


def test_run_inference_preserves_ids_and_emits_per_row_timing():
    loader = DataLoader(_TinyDataset(), batch_size=2, shuffle=False, collate_fn=_collate)

    probabilities, inference_seconds, aggregate_seconds = MODULE.run_inference(
        _TinyEncoder(),
        loader,
        ["spec-0", "spec-1", "spec-2"],
        device=torch.device("cpu"),
        fingerprint_bits=4,
        warmup_batches=1,
    )

    assert probabilities.shape == (3, 4)
    assert np.isfinite(probabilities).all()
    assert np.logical_and(probabilities >= 0.0, probabilities <= 1.0).all()
    assert inference_seconds.shape == (3,)
    assert np.all(inference_seconds >= 0.0)
    assert inference_seconds.sum() == pytest.approx(aggregate_seconds)


def test_state_dict_hash_is_key_order_independent():
    first = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
    second = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}

    assert MODULE.state_dict_sha256(first) == MODULE.state_dict_sha256(second)
