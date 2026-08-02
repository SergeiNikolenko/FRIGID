from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marlin.tokenizer import load_safe_tokenizer
from marlin.training import (
    MarlinCollator,
    MarlinSpectrumFingerprintDataset,
)


TOKENIZER_PATH = Path(
    "/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/"
    "data/safe-gpt/tokenizer.json"
)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        [
            {
                "spec_name": "spectrum-1",
                "smiles": "CCO",
                "inchikey_first_block": "LFQSCWFLJHTTHZ",
                "neutral_mass": 46.0401,
            },
            {
                "spec_name": "spectrum-2",
                "smiles": "CCN",
                "inchikey_first_block": "QUSNBJAOOMFDIB",
                "neutral_mass": 45.0578,
            },
        ]
    ).to_csv(metadata, index=False)
    fingerprints = tmp_path / "predictions.npz"
    values = np.zeros((2, 4096), dtype=np.float32)
    values[0, [2, 7]] = [0.94, 0.96]
    values[1, 9] = 0.99
    np.savez_compressed(
        fingerprints,
        probs=values,
        spectrum_ids=np.asarray(["spectrum-1", "spectrum-2"]),
    )
    return metadata, fingerprints


def test_spectrum_dataset_aligns_thresholds_and_excludes_selection_rows(tmp_path):
    metadata, fingerprints = _write_inputs(tmp_path)
    selection = tmp_path / "selection.csv"
    pd.DataFrame(
        [{"inchikey_first_block": "QUSNBJAOOMFDIB"}]
    ).to_csv(selection, index=False)
    tokenizer = load_safe_tokenizer(TOKENIZER_PATH)

    dataset = MarlinSpectrumFingerprintDataset(
        metadata,
        fingerprints,
        tokenizer,
        fingerprint_key="probs",
        threshold=0.95,
        max_length=256,
        exclude_metadata_csvs=(selection,),
    )

    assert len(dataset) == 1
    assert dataset[0]["spec_name"] == "spectrum-1"
    assert dataset[0]["fingerprint"][2] == 0
    assert dataset[0]["fingerprint"][7] == 1


def test_collator_uses_provided_fingerprint_and_spectrum_mass(tmp_path):
    metadata, fingerprints = _write_inputs(tmp_path)
    tokenizer = load_safe_tokenizer(TOKENIZER_PATH)
    dataset = MarlinSpectrumFingerprintDataset(
        metadata,
        fingerprints,
        tokenizer,
        fingerprint_key="probs",
        threshold=0.95,
        max_length=256,
    )

    batch = MarlinCollator(
        tokenizer,
        max_length=256,
        fingerprint_bits=4096,
    )([dataset[0]])

    assert batch["fingerprint"][0, 2].item() == 0
    assert batch["fingerprint"][0, 7].item() == 1
    assert batch["precursor_mass"][0].item() == pytest.approx(46.0401)


def test_spectrum_dataset_can_preserve_probability_amplitudes(tmp_path):
    metadata, fingerprints = _write_inputs(tmp_path)
    tokenizer = load_safe_tokenizer(TOKENIZER_PATH)
    dataset = MarlinSpectrumFingerprintDataset(
        metadata,
        fingerprints,
        tokenizer,
        fingerprint_key="probs",
        threshold=0.5,
        max_length=256,
        preserve_probabilities=True,
    )

    assert dataset[0]["fingerprint"][2] == pytest.approx(0.94)
    assert dataset[0]["fingerprint"][7] == pytest.approx(0.96)


def test_spectrum_dataset_requires_unique_spectrum_ids(tmp_path):
    metadata, _ = _write_inputs(tmp_path)
    fingerprints = tmp_path / "duplicate.npz"
    np.savez_compressed(
        fingerprints,
        probs=np.zeros((2, 4096), dtype=np.float32),
        spectrum_ids=np.asarray(["spectrum-1", "spectrum-1"]),
    )
    tokenizer = load_safe_tokenizer(TOKENIZER_PATH)

    with pytest.raises(ValueError, match="missing or duplicated"):
        MarlinSpectrumFingerprintDataset(
            metadata,
            fingerprints,
            tokenizer,
            fingerprint_key="probs",
            threshold=0.95,
            max_length=256,
        )
