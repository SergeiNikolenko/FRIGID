"""Load the exact 1,880-token SAFE vocabulary used by MARLIN and FRIGID."""

from __future__ import annotations

import json
from pathlib import Path

from safe.tokenizer import SAFETokenizer


def load_safe_tokenizer(path: str | Path):
    """Load a SAFE tokenizer export, including its custom SAFE pre-tokenizer."""
    with Path(path).open() as handle:
        payload = json.load(handle)
    return SAFETokenizer.from_dict(payload).get_pretrained()
