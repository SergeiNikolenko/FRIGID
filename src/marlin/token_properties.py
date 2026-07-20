"""Derive conservative heavy-atom properties for SAFE vocabulary tokens."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable

from rdkit import Chem


_PERIODIC_TABLE = Chem.GetPeriodicTable()
_HEAVY_ELEMENTS = tuple(
    _PERIODIC_TABLE.GetElementSymbol(atomic_number) for atomic_number in range(2, 119)
)
ATOM_PATTERN = re.compile(
    "|".join(
        re.escape(symbol)
        for symbol in sorted(
            (*_HEAVY_ELEMENTS, "b", "c", "n", "o", "p", "s"), key=len, reverse=True
        )
    )
)


@dataclass(frozen=True)
class TokenProperties:
    heavy_mass: float
    heavy_atoms: int
    valence_sum: float


def token_properties(token: str) -> TokenProperties:
    """Return a lower-bound mass and a conservative valence budget."""
    symbols = [
        match.capitalize() if len(match) == 1 else match
        for match in ATOM_PATTERN.findall(token)
    ]
    mass = 0.0
    valence = 0.0
    for symbol in symbols:
        atomic_number = _PERIODIC_TABLE.GetAtomicNumber(symbol)
        mass += _PERIODIC_TABLE.GetMostCommonIsotopeMass(atomic_number)
        valences = list(_PERIODIC_TABLE.GetValenceList(atomic_number))
        valence += max(valences) if valences else 0.0
    return TokenProperties(mass, len(symbols), valence)


def build_token_property_table(
    vocabulary_size: int,
    decode_token: Callable[[int], str],
    special_token_ids: Iterable[int] = (),
) -> tuple[list[float], list[int], list[float]]:
    special = set(special_token_ids)
    masses: list[float] = []
    atom_counts: list[int] = []
    valences: list[float] = []
    for token_id in range(vocabulary_size):
        properties = (
            TokenProperties(0.0, 0, 0.0)
            if token_id in special
            else token_properties(decode_token(token_id))
        )
        masses.append(properties.heavy_mass)
        atom_counts.append(properties.heavy_atoms)
        valences.append(properties.valence_sum)
    return masses, atom_counts, valences
