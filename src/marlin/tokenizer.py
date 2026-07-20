"""Load the exact 1,880-token SAFE vocabulary used by MARLIN and FRIGID."""

from __future__ import annotations

import json
from pathlib import Path

def load_safe_tokenizer(path: str | Path):
    """Load a SAFE tokenizer export, including its custom SAFE pre-tokenizer."""
    from safe.tokenizer import SAFETokenizer

    with Path(path).open() as handle:
        payload = json.load(handle)
    return SAFETokenizer.from_dict(payload).get_pretrained()


def validate_safe_tokenizer(
    tokenizer,
    *,
    expected_vocab_size: int,
    expected_special_token_ids: dict[str, int],
) -> dict[str, int]:
    """Fail closed when the tokenizer contract differs from the trained decoder."""
    actual = {
        "unk": tokenizer.unk_token_id,
        "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id,
        "pad": tokenizer.pad_token_id,
        "mask": tokenizer.mask_token_id,
    }
    if len(tokenizer) != expected_vocab_size:
        raise ValueError(
            f"SAFE vocabulary has {len(tokenizer)} tokens; "
            f"expected {expected_vocab_size}"
        )
    if actual != expected_special_token_ids:
        raise ValueError(
            f"SAFE special-token IDs are {actual}; "
            f"expected {expected_special_token_ids}"
        )
    return actual
