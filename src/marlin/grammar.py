"""Conservative lexical grammar mask for incremental SAFE/SMILES decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch


_TWO_CHARACTER_ATOMS = ("Br", "Cl")
_ONE_CHARACTER_ATOMS = frozenset("BCNOPSFIbcnosp*")
_BONDS = frozenset("-=#:/\\~")


@dataclass
class _GrammarState:
    expect_atom: bool = True
    branch_depth: int = 0
    atom_index: int = -1
    open_rings: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.open_rings is None:
            self.open_rings = {}

    @property
    def terminal(self) -> bool:
        return not self.expect_atom and self.branch_depth == 0 and not self.open_rings


def _scan(text: str) -> _GrammarState | None:
    state = _GrammarState()
    index = 0
    while index < len(text):
        char = text[index]
        if char == "[":
            close = text.find("]", index + 1)
            if close < 0 or not state.expect_atom:
                return None
            state.atom_index += 1
            state.expect_atom = False
            index = close + 1
            continue
        if text.startswith(_TWO_CHARACTER_ATOMS, index):
            if not state.expect_atom:
                state.expect_atom = True
            state.atom_index += 1
            state.expect_atom = False
            index += 2
            continue
        if char in _ONE_CHARACTER_ATOMS:
            if not state.expect_atom:
                state.expect_atom = True
            state.atom_index += 1
            state.expect_atom = False
            index += 1
            continue
        if char in _BONDS:
            if state.expect_atom:
                return None
            state.expect_atom = True
            index += 1
            continue
        if char == "(":
            if state.expect_atom:
                return None
            state.branch_depth += 1
            state.expect_atom = True
            index += 1
            continue
        if char == ")":
            if state.expect_atom or state.branch_depth == 0:
                return None
            state.branch_depth -= 1
            state.expect_atom = False
            index += 1
            continue
        if char == ".":
            if state.expect_atom:
                return None
            state.expect_atom = True
            index += 1
            continue
        if char == "%":
            if index + 2 >= len(text) or not text[index + 1 : index + 3].isdigit():
                return None
            label = text[index : index + 3]
            index += 3
        elif char.isdigit():
            label = char
            index += 1
        else:
            return None
        if state.expect_atom:
            return None
        assert state.open_rings is not None
        opening_atom = state.open_rings.get(label)
        if opening_atom is None:
            state.open_rings[label] = state.atom_index
        elif opening_atom == state.atom_index:
            return None
        else:
            del state.open_rings[label]
    return state


class SafeGrammarMask:
    """Mask higher-scoring tokens until the best lexically viable token remains."""

    def __init__(
        self,
        token_strings: Sequence[str],
        decode_prefix: Callable[[Sequence[int]], str],
        *,
        eos_token_id: int,
        special_token_ids: Sequence[int],
    ) -> None:
        self.token_strings = tuple(token_strings)
        self.decode_prefix = decode_prefix
        self.eos_token_id = eos_token_id
        self.special_token_ids = frozenset(special_token_ids)

    def __call__(self, prefix_ids: Sequence[int], logits: torch.Tensor) -> torch.Tensor:
        prefix = self.decode_prefix(prefix_ids)
        state = _scan(prefix)
        if state is None:
            return torch.full_like(logits, -torch.inf)
        constrained = logits.clone()
        while True:
            score, token = constrained.max(dim=-1)
            if not torch.isfinite(score):
                return constrained
            token_id = int(token)
            if token_id == self.eos_token_id:
                valid = state.terminal
            elif token_id in self.special_token_ids:
                valid = False
            else:
                valid = _scan(prefix + self.token_strings[token_id]) is not None
            if valid:
                return constrained
            constrained[token_id] = -torch.inf
