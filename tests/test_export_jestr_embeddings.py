import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "export_jestr_embeddings.py"
SPEC = importlib.util.spec_from_file_location("export_jestr_embeddings", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_spectrum(
    path: Path,
    *,
    ionization: str = "[M+H]+",
    peaks: list[tuple[float, float]] | None = None,
) -> None:
    peaks = peaks or [(50.0, 1.0), (75.0, 0.5)]
    peak_rows = "\n".join(f"{mz} {intensity}" for mz, intensity in peaks)
    path.write_text(
        f">compound example\n"
        f">parentmass 100.0\n"
        f">ionization {ionization}\n"
        f"\n>ms2peaks\n{peak_rows}\n"
    )


def released_reference(peaks: list[tuple[float, float]]) -> np.ndarray:
    maximum = max(intensity for _, intensity in peaks)
    internal_rows = [
        (intensity * MODULE.MAX_NORMALIZED_INTENSITY / maximum, mz)
        for mz, intensity in peaks
    ]
    output = np.zeros(MODULE.BIN_COUNT, dtype=np.float32)
    for intensity, mz in internal_rows:
        if mz < MODULE.MAX_MZ + MODULE.BIN_RESOLUTION:
            output[int((mz - 1) / MODULE.BIN_RESOLUTION)] += intensity
    return np.log10(output + 1) / 3


def test_prepare_spectrum_matches_released_preprocessing_exactly(tmp_path: Path):
    spectrum_path = tmp_path / "example.ms"
    peaks = [
        (1.5, 2.0),
        (1.7, 1.0),
        (2.2, 4.0),
        (1000.5, 2.0),
        (1001.0, 8.0),
    ]
    write_spectrum(spectrum_path, ionization="[M+Na]+", peaks=peaks)

    prepared = MODULE.prepare_spectrum(spectrum_path)

    np.testing.assert_array_equal(prepared.binned, released_reference(peaks))
    assert prepared.binned.shape == (1000,)
    assert prepared.binned.dtype == np.float32
    assert prepared.ionization == "[M+Na]+"
    assert prepared.source_peak_count == 5
    assert prepared.retained_peak_count == 4
    assert prepared.removed_above_max_mz_count == 1
    assert prepared.source_max_intensity == 8.0


def test_prepare_spectrum_rejects_all_zero_intensities(tmp_path: Path):
    spectrum_path = tmp_path / "zeros.ms"
    write_spectrum(spectrum_path, peaks=[(10.0, 0.0), (20.0, 0.0)])

    with pytest.raises(ValueError, match="no positive peak intensity"):
        MODULE.prepare_spectrum(spectrum_path)


def test_metadata_is_sorted_and_inchikey_column_is_resolved(tmp_path: Path):
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "fingerprint_index": [1, 0],
            "spec_name": ["b", "a"],
            "inchi_key": ["BBBBBBBBBBBBBB", "AAAAAAAAAAAAAA"],
        }
    ).to_csv(metadata_path, index=False)

    metadata, inchikey_column = MODULE.load_ordered_metadata(
        metadata_path,
        id_column="spec_name",
        index_column="fingerprint_index",
        inchikey_column=None,
    )

    assert metadata["spec_name"].tolist() == ["a", "b"]
    assert inchikey_column == "inchi_key"


def test_main_writes_complete_row_locked_bundle(tmp_path: Path, monkeypatch):
    torch = pytest.importorskip("torch")

    class FakeModel:
        def __call__(self, binned):
            return binned[:, : MODULE.EMBEDDING_DIMENSION]

    def fake_load_model(*args, **kwargs):
        return FakeModel(), torch, "cpu", "test-revision", "test-checkpoint-sha256"

    def fake_targets(metadata, **kwargs):
        targets = np.zeros((len(metadata), MODULE.FINGERPRINT_BITS), dtype=np.uint8)
        targets[:, 0] = 1
        return targets

    monkeypatch.setattr(MODULE, "_load_official_model", fake_load_model)
    monkeypatch.setattr(MODULE, "compute_morgan_targets", fake_targets)

    spectra_dir = tmp_path / "spectra"
    spectra_dir.mkdir()
    write_spectrum(spectra_dir / "spec-a.ms", peaks=[(10.0, 1.0)])
    write_spectrum(
        spectra_dir / "spec-b.ms",
        ionization="[M+Na]+",
        peaks=[(20.0, 1.0)],
    )
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "fingerprint_index": [1, 0],
            "spec_name": ["spec-b", "spec-a"],
            "inchikey": ["BBBBBBBBBBBBBB", "AAAAAAAAAAAAAA"],
            "smiles": ["CC", "C"],
        }
    ).to_csv(metadata_path, index=False)
    output_path = tmp_path / "bundle.npz"

    exit_code = MODULE.main(
        [
            "--metadata",
            str(metadata_path),
            "--spectra-dir",
            str(spectra_dir),
            "--jestr-repo",
            str(tmp_path / "repo"),
            "--output",
            str(output_path),
            "--compute-morgan-targets",
            "--device",
            "cpu",
            "--batch-size",
            "2",
        ]
    )

    assert exit_code == 0
    with np.load(output_path, allow_pickle=False) as bundle:
        assert set(bundle.files) == {
            "aggregate_inference_seconds",
            "embeddings",
            "fingerprint_index",
            "ground_truth",
            "inchikeys",
            "inference_seconds",
            "spectrum_ids",
        }
        assert bundle["embeddings"].shape == (2, 512)
        assert bundle["ground_truth"].shape == (2, 4096)
        assert bundle["ground_truth"].dtype == np.uint8
        assert bundle["spectrum_ids"].tolist() == ["spec-a", "spec-b"]
        assert bundle["inchikeys"].tolist() == ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"]
        assert bundle["inference_seconds"].shape == (2,)
        assert bundle["inference_seconds"].sum() == pytest.approx(
            float(bundle["aggregate_inference_seconds"])
        )

    manifest = json.loads(output_path.with_suffix(".manifest.json").read_text())
    assert manifest["model"]["source_revision"] == "test-revision"
    assert manifest["coverage"]["ionization_counts"] == {
        "[M+H]+": 1,
        "[M+Na]+": 1,
    }
    assert manifest["outputs"]["npz_keys"] == sorted(
        [
            "aggregate_inference_seconds",
            "embeddings",
            "fingerprint_index",
            "ground_truth",
            "inchikeys",
            "inference_seconds",
            "spectrum_ids",
        ]
    )
