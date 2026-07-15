import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "export_msbert_embeddings.py"
SPEC = importlib.util.spec_from_file_location("export_msbert_embeddings", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_spectrum(path: Path, *, ionization: str = "[M+H]+", peaks: list[tuple] | None = None):
    peaks = peaks or [(10.004, 1.0), (12.345, 0.5)]
    peak_rows = "\n".join(f"{mz} {intensity}" for mz, intensity in peaks)
    path.write_text(
        f">compound example\n"
        f">parentmass 100.004\n"
        f">ionization {ionization}\n"
        f"\n>ms2peaks\n{peak_rows}\n"
    )


def test_prepare_spectrum_matches_release_tokenization_and_domain_filter(tmp_path):
    spectrum_path = tmp_path / "example.ms"
    write_spectrum(
        spectrum_path,
        peaks=[
            (5.0, 99.0),
            (10.004, 1.0),
            (12.345, 0.5),
            (999.994, 3.0),
            (1000.1, 99.0),
        ],
    )
    vocabulary = MODULE.build_released_vocabulary()

    prepared = MODULE.prepare_spectrum(spectrum_path, vocabulary)

    expected_words = [f"{100.004:.2f}", f"{10.004:.2f}", f"{12.345:.2f}", f"{999.994:.2f}"]
    assert prepared.input_ids[:4].tolist() == [vocabulary[word] for word in expected_words]
    assert prepared.input_ids.shape == (100,)
    assert prepared.intensity.shape == (100,)
    assert prepared.intensity[:4].tolist() == pytest.approx([2 / 3, 1 / 3, 1 / 6, 1])
    assert np.count_nonzero(prepared.input_ids[4:]) == 0
    assert prepared.source_peak_count == 5
    assert prepared.retained_peak_count == 3
    assert prepared.removed_domain_peak_count == 2
    assert prepared.truncated_peak_count == 0


def test_prepare_spectrum_keeps_top_99_peaks_in_original_order(tmp_path):
    spectrum_path = tmp_path / "many.ms"
    peaks = [(20.0 + index * 0.01, float(index)) for index in range(101)]
    write_spectrum(spectrum_path, peaks=peaks)
    vocabulary = MODULE.build_released_vocabulary()

    prepared = MODULE.prepare_spectrum(spectrum_path, vocabulary)

    expected_words = [f"{mz:.2f}" for mz, _ in peaks[2:]]
    assert prepared.input_ids[1:].tolist() == [vocabulary[word] for word in expected_words]
    assert prepared.retained_peak_count == 99
    assert prepared.truncated_peak_count == 2


def test_prepare_spectrum_rejects_negative_ion_mode(tmp_path):
    spectrum_path = tmp_path / "negative.ms"
    write_spectrum(spectrum_path, ionization="[M-H]-")

    with pytest.raises(ValueError, match="positive ion mode"):
        MODULE.prepare_spectrum(spectrum_path, MODULE.build_released_vocabulary())


def test_metadata_is_sorted_and_validated_by_fingerprint_index(tmp_path):
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "fingerprint_index": [1, 0],
            "spec_name": ["b", "a"],
            "inchi_key": ["BBBBBBBBBBBBBB-X", "AAAAAAAAAAAAAA-X"],
        }
    ).to_csv(metadata_path, index=False)

    metadata = MODULE.load_ordered_metadata(
        metadata_path,
        id_column="spec_name",
        index_column="fingerprint_index",
    )

    assert metadata["spec_name"].tolist() == ["a", "b"]
