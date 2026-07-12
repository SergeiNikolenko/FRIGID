import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_msg_compact_benchmark.py"
SPEC = importlib.util.spec_from_file_location(
    "build_msg_compact_benchmark", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_fixture(tmp_path: Path, rows: int = 48):
    metadata_rows = []
    label_rows = []
    fingerprints = np.zeros((rows, 16), dtype=np.float32)
    spectrum_dir = tmp_path / "spec"
    spectrum_dir.mkdir()
    for index in range(rows):
        spec_name = f"q{index:03d}"
        molecule = f"M{index // 2:04d}"
        formula = f"C{6 + index % 7}H{10 + index % 9}N{1 + index % 2}"
        metadata_rows.append(
            {
                "fingerprint_index": index,
                "spec_name": spec_name,
                "inchi_key_first_block": molecule,
            }
        )
        label_rows.append(
            {
                "spec": spec_name,
                "formula": formula,
                "inchikey": molecule,
                "instrument": "Orbitrap" if index % 4 else "QTOF",
                "ionization": "[M+Na]+" if index % 5 == 0 else "[M+H]+",
            }
        )
        fingerprints[index, : 2 + index % 8] = 1
        peaks = "\n".join(
            f"{100 + peak}.0 {1 / (peak + 1):.4f}" for peak in range(2 + index % 10)
        )
        (spectrum_dir / f"{spec_name}.ms").write_text(
            f">compound {spec_name}\n>parentmass {250 + index * 4}.0\n\n>ms2peaks\n{peaks}\n"
        )
    metadata_path = tmp_path / "metadata.csv"
    labels_path = tmp_path / "labels.tsv"
    fingerprints_path = tmp_path / "fingerprints.npz"
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    pd.DataFrame(label_rows).to_csv(labels_path, sep="\t", index=False)
    np.savez(fingerprints_path, mist_binary=fingerprints)
    exclusions = []
    for offset, size in ((0, 2), (2, 2)):
        path = tmp_path / f"exclude_{offset}.tsv"
        pd.DataFrame(
            {"spec_name": [f"q{index:03d}" for index in range(offset, offset + size)]}
        ).to_csv(path, sep="\t", index=False)
        exclusions.append(path)
    return metadata_path, labels_path, fingerprints_path, spectrum_dir, exclusions


def test_compact_panels_are_deterministic_nested_and_disjoint(tmp_path):
    inputs = _write_fixture(tmp_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    kwargs = {
        "metadata_csv": inputs[0],
        "labels_tsv": inputs[1],
        "fingerprints_npz": inputs[2],
        "spectrum_dir": inputs[3],
        "exclusion_manifests": inputs[4],
        "expected_population_size": 48,
        "micro_sizes": (8, 12, 16),
        "macro_size": 4,
    }
    first = MODULE.build_compact_benchmark(output_dir=first_dir, **kwargs)
    MODULE.build_compact_benchmark(output_dir=second_dir, **kwargs)

    micro8 = pd.read_csv(first_dir / "msg_compact_micro8_v1.tsv", sep="\t")
    micro12 = pd.read_csv(first_dir / "msg_compact_micro12_v1.tsv", sep="\t")
    micro16 = pd.read_csv(first_dir / "msg_compact_micro16_v1.tsv", sep="\t")
    macro = pd.read_csv(first_dir / "msg_compact_macro64_v1.tsv", sep="\t")
    repeated = pd.read_csv(second_dir / "msg_compact_micro16_v1.tsv", sep="\t")

    assert micro16["spec_name"].tolist() == repeated["spec_name"].tolist()
    assert micro8["spec_name"].tolist() == micro12["spec_name"].tolist()[:8]
    assert micro12["spec_name"].tolist() == micro16["spec_name"].tolist()[:12]
    assert set(micro16["spec_name"]).isdisjoint(macro["spec_name"])
    assert set(micro16["inchikey_first_block"]).isdisjoint(
        macro["inchikey_first_block"]
    )
    assert first["quality_checks"]["labels_join_coverage"] == 1.0
    assert first["quality_checks"]["target_fields_used_for_model_scoring"] == []


def test_population_join_fails_closed_on_missing_label(tmp_path):
    inputs = _write_fixture(tmp_path, rows=16)
    labels = pd.read_csv(inputs[1], sep="\t").iloc[:-1]
    labels.to_csv(inputs[1], sep="\t", index=False)
    with pytest.raises(ValueError, match="lack labels"):
        MODULE.build_population_frame(
            metadata_csv=inputs[0],
            labels_tsv=inputs[1],
            fingerprints_npz=inputs[2],
            spectrum_dir=inputs[3],
            expected_population_size=16,
        )


def test_formula_parser_tracks_heavy_atoms_and_element_flags():
    features = MODULE.formula_features("C12H16BrN2O2PS")
    assert features == {
        "formula_heavy_atoms": 19,
        "has_phosphorus": 1,
        "has_sulfur": 1,
        "has_halogen": 1,
    }
    assert MODULE.formula_features("C22H28N7O+")["formula_heavy_atoms"] == 30
