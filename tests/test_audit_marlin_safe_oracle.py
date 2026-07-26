from pathlib import Path

import safe

from marlin.tokenizer import load_safe_tokenizer
from marlin.training import MarlinCollator, MarlinMetadataDataset
from scripts.audit_marlin_safe_oracle import encode_audit_sequence


TOKENIZER_PATH = Path(
    "/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/"
    "data/safe-gpt/tokenizer.json"
)


def test_audit_safe_sequence_matches_metadata_training_path(tmp_path):
    smiles = "CCOC(=O)c1ccccc1"
    tokenizer = load_safe_tokenizer(TOKENIZER_PATH)
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(f"smiles\n{smiles}\n")

    dataset = MarlinMetadataDataset(metadata, tokenizer, max_length=256)
    audited_safe, audited_token_ids = encode_audit_sequence(smiles, tokenizer)
    legacy_safe = safe.encode(
        smiles,
        canonical=True,
        randomize=False,
        ignore_stereo=True,
    )

    assert legacy_safe != dataset[0]["safe"]
    assert audited_safe == dataset[0]["safe"]

    batch = MarlinCollator(
        tokenizer,
        max_length=256,
        fingerprint_bits=16,
    )([dataset[0]])
    collated_token_ids = batch["input_ids"][0].tolist()

    assert audited_token_ids == collated_token_ids
    assert len(audited_token_ids) == int(batch["input_ids"].shape[1])
