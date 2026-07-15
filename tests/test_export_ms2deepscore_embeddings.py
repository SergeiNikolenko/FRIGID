from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_ms2deepscore_embeddings.py"
SPEC = importlib.util.spec_from_file_location("export_ms2deepscore_embeddings", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_spectrum(path: Path, ionization: str = "[M+H]+") -> None:
    path.write_text(
        "\n".join(
            [
                ">compound example",
                ">parentmass 250.125",
                f">ionization {ionization}",
                "",
                ">ms2peaks",
                "5.0 0.5",
                "10.0 2.0",
                "100.0 4.0",
                "999.9 1.0",
                "1000.0 3.0",
            ]
        )
    )


@pytest.mark.parametrize(
    ("ionization", "expected_mode"),
    [("[M+H]+", "positive"), ("[M-H]-", "negative")],
)
def test_parse_ms_file_applies_released_domain_and_normalization(
    tmp_path: Path,
    ionization: str,
    expected_mode: str,
) -> None:
    spectrum_path = tmp_path / "example.ms"
    _write_spectrum(spectrum_path, ionization)

    spectrum = MODULE.parse_ms_file(spectrum_path)

    assert spectrum.precursor_mz == pytest.approx(250.125)
    assert spectrum.ion_mode == expected_mode
    np.testing.assert_allclose(spectrum.mz, [10.0, 100.0, 999.9])
    np.testing.assert_allclose(spectrum.intensities, [0.5, 1.0, 0.25])
    assert spectrum.source_peak_count == 5
    assert spectrum.removed_domain_peak_count == 2


def test_parse_ms_file_rejects_ambiguous_ionization(tmp_path: Path) -> None:
    spectrum_path = tmp_path / "example.ms"
    _write_spectrum(spectrum_path, "unknown")

    with pytest.raises(ValueError, match="Unsupported ionization"):
        MODULE.parse_ms_file(spectrum_path)


def test_load_ordered_metadata_sorts_and_checks_contiguous_indexes(
    tmp_path: Path,
) -> None:
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "spec_name": ["second", "first"],
            "fingerprint_index": [1, 0],
            "inchi_key": ["BBBB", "AAAA"],
        }
    ).to_csv(metadata_path, index=False)

    metadata = MODULE.load_ordered_metadata(
        metadata_path,
        id_column="spec_name",
        index_column="fingerprint_index",
        inchikey_column="inchi_key",
    )

    assert metadata["spec_name"].tolist() == ["first", "second"]
    assert metadata["inchi_key"].tolist() == ["AAAA", "BBBB"]


def test_load_ordered_metadata_rejects_index_gap(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "spec_name": ["first", "second"],
            "fingerprint_index": [0, 2],
            "inchi_key": ["AAAA", "BBBB"],
        }
    ).to_csv(metadata_path, index=False)

    with pytest.raises(ValueError, match="every index"):
        MODULE.load_ordered_metadata(
            metadata_path,
            id_column="spec_name",
            index_column="fingerprint_index",
            inchikey_column="inchi_key",
        )
