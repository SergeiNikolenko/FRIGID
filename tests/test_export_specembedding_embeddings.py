from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_specembedding_embeddings.py"
SPEC = importlib.util.spec_from_file_location("export_specembedding_embeddings", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_spectrum(
    path: Path,
    peaks: list[tuple[float, float]],
    ionization: str = "[M+H]+",
) -> None:
    lines = [
        ">compound example",
        ">parentmass 250.125",
        f">ionization {ionization}",
        "",
        ">ms2peaks",
    ]
    lines.extend(f"{mz} {intensity}" for mz, intensity in peaks)
    path.write_text("\n".join(lines))


def test_prepare_spectrum_prepends_precursor_and_pads(tmp_path: Path) -> None:
    spectrum_path = tmp_path / "example.ms"
    _write_spectrum(spectrum_path, [(50.0, 2.0), (75.0, 4.0)])

    spectrum = MODULE.prepare_spectrum(spectrum_path)

    assert spectrum.mz.shape == (100,)
    assert spectrum.intensity.shape == (100,)
    assert spectrum.mask.shape == (100,)
    np.testing.assert_allclose(spectrum.mz[:3], [250.125, 50.0, 75.0])
    np.testing.assert_allclose(spectrum.intensity[:3], [2.0, 0.5, 1.0])
    assert not spectrum.mask[:3].any()
    assert spectrum.mask[3:].all()
    assert spectrum.source_peak_count == 2
    assert spectrum.retained_peak_count == 2
    assert spectrum.truncated_peak_count == 0


def test_prepare_spectrum_selects_top_99_and_restores_order(tmp_path: Path) -> None:
    spectrum_path = tmp_path / "example.ms"
    peaks = [(float(index + 1), float(index + 1)) for index in range(101)]
    _write_spectrum(spectrum_path, peaks)

    spectrum = MODULE.prepare_spectrum(spectrum_path)

    assert spectrum.source_peak_count == 101
    assert spectrum.retained_peak_count == 99
    assert spectrum.truncated_peak_count == 2
    np.testing.assert_allclose(spectrum.mz[1:4], [3.0, 4.0, 5.0])
    np.testing.assert_allclose(spectrum.mz[-3:], [99.0, 100.0, 101.0])
    assert spectrum.intensity[-1] == pytest.approx(1.0)
    assert not spectrum.mask.any()


def test_prepare_spectrum_rejects_negative_mode(tmp_path: Path) -> None:
    spectrum_path = tmp_path / "example.ms"
    _write_spectrum(spectrum_path, [(50.0, 1.0)], ionization="[M-H]-")

    with pytest.raises(ValueError, match="positive-ion"):
        MODULE.prepare_spectrum(spectrum_path)
