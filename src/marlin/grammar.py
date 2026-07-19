"""Conservative lexical grammar mask for incremental SAFE/SMILES decoding."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

import torch


_TWO_CHARACTER_ATOMS = ("Br", "Cl")
_ONE_CHARACTER_ATOMS = frozenset("BCNOPSFIbcnosp*")
_BONDS = frozenset("-=#:/\\~")
_BRACKET_ATOM = re.compile(
    r"^(?P<isotope>\d{0,3})"
    r"(?P<element>Cl|Br|Si|Se|Na|Li|Mg|Ca|Al|[BCNOPSFIK]|[bcnops*])"
    r"@{0,2}(?:H\d{0,2})?(?:[+-]{1,3}\d{0,2})?(?::\d*)?$"
)


def _partial_bracket_symbol(content: str) -> str | None:
    if not content or (content.isdigit() and len(content) <= 3):
        return ""
    match = _BRACKET_ATOM.fullmatch(content)
    if not match:
        return None
    element = match.group("element")
    if element == "N" and "+" in content[match.end("element") :]:
        return "N+"
    return element


@dataclass
class _GrammarState:
    expect_atom: bool = True
    allow_bond: bool = False
    allow_ring: bool = False
    branch_depth: int = 0
    atom_index: int = -1
    current_atom: int | None = None
    branch_atoms: list[int] | None = None
    open_rings: dict[str, int] | None = None
    bond_counts: dict[int, int] | None = None
    bond_limits: dict[int, int] | None = None
    incomplete_token: bool = False

    def __post_init__(self) -> None:
        if self.open_rings is None:
            self.open_rings = {}
        if self.branch_atoms is None:
            self.branch_atoms = []
        if self.bond_counts is None:
            self.bond_counts = {}
        if self.bond_limits is None:
            self.bond_limits = {}

    @property
    def terminal(self) -> bool:
        return (
            not self.expect_atom
            and self.branch_depth == 0
            and not self.open_rings
            and not self.incomplete_token
        )


def _scan(text: str) -> _GrammarState | None:
    state = _GrammarState()

    def add_atom(symbol: str) -> bool:
        previous_atom = state.current_atom
        state.atom_index += 1
        state.current_atom = state.atom_index
        state.bond_limits[state.atom_index] = {
            "H": 1,
            "F": 1,
            "Cl": 1,
            "Br": 1,
            "I": 1,
            "B": 3,
            "b": 3,
            "C": 4,
            "c": 3,
            "N": 3,
            "N+": 4,
            "n": 3,
            "O": 2,
            "o": 2,
            "P": 5,
            "p": 3,
            "S": 6,
            "s": 4,
        }.get(symbol, 4)
        state.bond_counts[state.atom_index] = 0
        if previous_atom is not None:
            state.bond_counts[previous_atom] += 1
            state.bond_counts[state.atom_index] += 1
            if state.bond_counts[previous_atom] > state.bond_limits[previous_atom]:
                return False
        state.expect_atom = False
        state.allow_bond = False
        state.allow_ring = False
        return True

    index = 0
    while index < len(text):
        char = text[index]
        if char == "[":
            close = text.find("]", index + 1)
            if close < 0:
                symbol = _partial_bracket_symbol(text[index + 1 :])
                if symbol is None:
                    return None
                if symbol and not add_atom(symbol):
                    return None
                state.incomplete_token = True
                return state
            symbol = _partial_bracket_symbol(text[index + 1 : close])
            if not symbol:
                return None
            if not add_atom(symbol):
                return None
            index = close + 1
            continue
        if text.startswith(_TWO_CHARACTER_ATOMS, index):
            if not state.expect_atom:
                state.expect_atom = True
            if not add_atom(text[index : index + 2]):
                return None
            index += 2
            continue
        if char in _ONE_CHARACTER_ATOMS:
            if not state.expect_atom:
                state.expect_atom = True
            if not add_atom(char):
                return None
            index += 1
            continue
        if char in _BONDS:
            if state.expect_atom and not state.allow_bond:
                return None
            if (
                not state.expect_atom
                and state.current_atom is not None
                and state.bond_counts[state.current_atom]
                >= state.bond_limits[state.current_atom]
            ):
                return None
            state.allow_ring = not state.expect_atom
            state.expect_atom = True
            state.allow_bond = False
            index += 1
            continue
        if char == "(":
            if state.expect_atom or state.current_atom is None:
                return None
            if (
                state.bond_counts[state.current_atom]
                >= state.bond_limits[state.current_atom]
            ):
                return None
            state.branch_atoms.append(state.current_atom)
            state.branch_depth += 1
            state.expect_atom = True
            state.allow_bond = True
            state.allow_ring = False
            index += 1
            continue
        if char == ")":
            if state.expect_atom or state.branch_depth == 0:
                return None
            state.current_atom = state.branch_atoms.pop()
            state.branch_depth -= 1
            state.expect_atom = False
            state.allow_bond = False
            state.allow_ring = False
            index += 1
            continue
        if char == ".":
            if state.expect_atom:
                return None
            state.current_atom = None
            state.expect_atom = True
            state.allow_bond = False
            state.allow_ring = False
            index += 1
            continue
        if char == "%":
            remaining = text[index + 1 :]
            if len(remaining) < 2:
                if remaining.isdigit() or not remaining:
                    state.incomplete_token = True
                    return state
                return None
            if not text[index + 1 : index + 3].isdigit():
                return None
            label = text[index : index + 3]
            index += 3
        elif char.isdigit():
            label = char
            index += 1
        else:
            return None
        if state.expect_atom and not state.allow_ring:
            return None
        if state.current_atom is None:
            return None
        state.expect_atom = False
        state.allow_ring = False
        assert state.open_rings is not None
        opening_atom = state.open_rings.get(label)
        if opening_atom is None:
            state.open_rings[label] = state.current_atom
            atom = state.current_atom
        elif opening_atom == state.current_atom:
            return None
        else:
            del state.open_rings[label]
            atom = state.current_atom
        state.bond_counts[atom] += 1
        if state.bond_counts[atom] > state.bond_limits[atom]:
            return None
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
