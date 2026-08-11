"""Derive conservative heavy-atom properties for SAFE vocabulary tokens."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Iterable

from rdkit import Chem


HYDROGEN_MASS = 1.00782503223
PROTON_MASS = 1.007276466621
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
BRACKET_ATOM_PATTERN = re.compile(
    r"\[(?P<isotope>\d+)?(?P<symbol>[A-Z][a-z]?|[bcnops])(?P<annotations>[^]]*)\]?"
)
CHARGE_PATTERN = re.compile(r"(?P<signs>[+-]+)(?P<magnitude>\d*)")
ISOTOPE_TOKEN_PATTERN = re.compile(r"^\[\d")


def isotope_token_ids(token_strings: Iterable[str]) -> tuple[int, ...]:
    """Return vocabulary ids whose token opens a bracket atom with a mass number.

    The mass shell is defined on monoisotopic mass, so an isotopologue can only
    ever miss it; the pattern also covers the partial tokens that can lead
    nowhere else. On the NPLIB1 adaptation targets these account for 479 of the
    1,880 vocabulary entries and are used by none of the 7,144 gold answers.
    """
    return tuple(
        index
        for index, token in enumerate(token_strings)
        if token and ISOTOPE_TOKEN_PATTERN.match(token)
    )


@dataclass(frozen=True)
class TokenProperties:
    heavy_mass: float
    heavy_atoms: int
    valence_sum: float


def element_symbol(symbol: str) -> str | None:
    """Canonicalise an atom symbol, or ``None`` when it names no element.

    The wildcard "*" is the symbol SAFE can write that names no element, and so
    has no mass; aromatic atoms are written in lower case and are the same
    element as their upper-case form.
    """
    canonical = symbol.capitalize() if len(symbol) == 1 else symbol
    try:
        return canonical if _PERIODIC_TABLE.GetAtomicNumber(canonical) > 0 else None
    except RuntimeError:
        return None


@lru_cache(maxsize=None)
def atom_mass(element: str, mass_number: int | None = None) -> float | None:
    """Return the mass of one atom of ``element``, or ``None`` if there is none.

    This is the single place that answers "what does this atom weigh". The
    grammar used to keep a private 18-element table with no hydrogen entry and
    weighed everything outside it at 0.0, while this module charged the same
    characters their real mass: measured over the 803-spectrum run, 2,222 of
    3,093 dead-end prefixes (71.8%) had the two models disagreeing by more than
    0.01 Da, with a median 27.0 Da gap on the prefixes the token-table prune
    killed. One function removes the disagreement by construction.

    ``mass_number`` selects an isotope and returns ``None`` when the element has
    no isotope at that mass, which is how an impossible label is refused rather
    than silently taking the element's most common mass.
    """
    canonical = element_symbol(element)
    if canonical is None:
        return None
    atomic_number = _PERIODIC_TABLE.GetAtomicNumber(canonical)
    if mass_number is None:
        return _PERIODIC_TABLE.GetMostCommonIsotopeMass(atomic_number)
    mass = _PERIODIC_TABLE.GetMassForIsotope(atomic_number, mass_number)
    return mass if mass > 0 else None


def bracket_charge(annotations: str) -> int:
    """Return the formal charge written after the element inside a bracket atom."""
    match = CHARGE_PATTERN.search(annotations)
    if match is None:
        return 0
    sign = 1 if match.group("signs")[0] == "+" else -1
    magnitude = match.group("magnitude")
    return sign * (int(magnitude) if magnitude else len(match.group("signs")))


def _token_atoms(token: str) -> list[tuple[str, int | None, int]]:
    """Return the (element, mass number, formal charge) triples of a token."""
    bracket_atoms = list(BRACKET_ATOM_PATTERN.finditer(token))
    bracket_ranges = [match.span() for match in bracket_atoms]
    atoms: list[tuple[str, int | None, int]] = []
    for match in bracket_atoms:
        symbol = element_symbol(match.group("symbol"))
        if symbol is None:
            continue
        atoms.append(
            (
                symbol,
                int(match.group("isotope")) if match.group("isotope") else None,
                bracket_charge(match.group("annotations")),
            )
        )
    for match in ATOM_PATTERN.finditer(token):
        if any(start <= match.start() < end for start, end in bracket_ranges):
            continue
        symbol = match.group()
        # SMILES cannot write a charge outside a bracket, so these are neutral.
        atoms.append((symbol.capitalize() if len(symbol) == 1 else symbol, None, 0))
    return atoms


ORGANIC_ELEMENTS = frozenset({"C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I"})


def foreign_element_token_ids(
    token_strings: Iterable[str],
    allowed_elements: Iterable[str] = ORGANIC_ELEMENTS,
) -> tuple[int, ...]:
    """Return vocabulary ids that introduce an element outside ``allowed_elements``.

    The default set is the one small-molecule MS structure elucidation works in,
    declared from the assay rather than from the answers. The vocabulary
    inherited from the pretraining corpus can write 123 distinct elements, of
    which 1,014 token entries fall outside this set; the 7,144 NPLIB1 adaptation
    targets between them use only C, N, O, P, S, F and Cl.

    A token whose element cannot be resolved is left supported, so the filter
    never removes more than it can justify.
    """
    allowed = frozenset(allowed_elements)
    return tuple(
        index
        for index, token in enumerate(token_strings)
        if token
        and (symbols := {symbol for symbol, _, _ in _token_atoms(token)})
        and not symbols <= allowed
    )


def token_properties(token: str) -> TokenProperties:
    """Return a lower-bound mass and a conservative valence budget.

    The mass is stated in the convention the conditioning mass is stated in: the
    run derives its target as ``precursor_mz - proton``, so a formal charge
    written into the token costs one hydrogen atom mass, the proton plus the
    electron the ion is missing. Without that, a candidate carrying its own
    charge is compared against a target short by 1.0073 Da, which is 1,269 to
    5,354 ppm against a 10 ppm window.
    """
    atoms = _token_atoms(token)
    mass = 0.0
    valence = 0.0
    for symbol, isotope, charge in atoms:
        atomic_number = _PERIODIC_TABLE.GetAtomicNumber(symbol)
        # An impossible isotope label stays conservative at 0.0 instead of
        # silently taking the element's most-common isotope mass.
        mass += (atom_mass(symbol, isotope) or 0.0) - charge * HYDROGEN_MASS
        valences = list(_PERIODIC_TABLE.GetValenceList(atomic_number))
        valence += max(valences) if valences else 0.0
    return TokenProperties(mass, len(atoms), valence)


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
