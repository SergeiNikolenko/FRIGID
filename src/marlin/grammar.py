"""Conservative lexical grammar mask for incremental SAFE/SMILES decoding."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor
from typing import Callable, NamedTuple, Sequence

import numpy as np
import torch
from rdkit import Chem


_TWO_CHARACTER_ATOMS = ("Br", "Cl")
_TWO_CHARACTER_ATOM_STARTS = frozenset(atom[0] for atom in _TWO_CHARACTER_ATOMS)
_ONE_CHARACTER_ATOMS = frozenset("BCNOPSFIbcnosp*")
_PERIODIC_TABLE = Chem.GetPeriodicTable()
_ELEMENT_PATTERN = "|".join(
    re.escape(_PERIODIC_TABLE.GetElementSymbol(atomic_number))
    for atomic_number in sorted(
        range(1, 119),
        key=lambda value: len(_PERIODIC_TABLE.GetElementSymbol(value)),
        reverse=True,
    )
)
_BONDS = frozenset("-=#:/\\~")
_BOND_ORDERS = {"-": 1.0, "=": 2.0, "#": 3.0, ":": 1.5, "/": 1.0, "\\": 1.0, "~": 1.0}
_HYDROGEN_MASS = 1.00782503223
_ATOM_MASSES = {
    "B": 11.00930536,
    "C": 12.0,
    "N": 14.003074004,
    "O": 15.99491462,
    "F": 18.998403163,
    "P": 30.973761998,
    "S": 31.972071174,
    "Cl": 34.96885268,
    "K": 38.963706486,
    "Br": 78.9183376,
    "I": 126.904468,
    "Si": 27.976926535,
    "Se": 79.9165218,
    "Na": 22.989769282,
    "Li": 7.016003437,
    "Mg": 23.985041697,
    "Ca": 39.962590863,
    "Al": 26.98153853,
}
_ATOM_VALENCES = {
    "B": 3,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "P": 5,
    "S": 6,
    "Cl": 1,
    "K": 1,
    "Br": 1,
    "I": 1,
    "Si": 4,
    "Se": 6,
    "Na": 1,
    "Li": 1,
    "Mg": 2,
    "Ca": 2,
    "Al": 3,
}
_REACHABILITY_SCALE = 1_000
_BRACKET_COMPLETION_ELEMENTS = tuple(_ATOM_MASSES)
_ATOM_COMPLETIONS = (
    "B",
    "C",
    "N",
    "O",
    "P",
    "S",
    "F",
    "I",
    "Cl",
    "Br",
    "c",
    "n",
    "o",
    "p",
    "s",
)
_BRACKET_ATOM = re.compile(
    r"^(?P<isotope>\d{0,3})"
    rf"(?P<element>{_ELEMENT_PATTERN}|[bcnops*])"
    r"@{0,2}(?:H\d{0,2})?(?:[+-]{1,3}\d{0,2})?(?::\d*)?$"
)


@lru_cache(maxsize=None)
def _rdkit_valences(symbol: str) -> tuple[int, tuple[int, ...]]:
    """Return RDKit's default valence and allowed valence list for ``symbol``.

    ``(-1, ())`` marks a symbol RDKit gives no valence model, which is the
    wildcard "*" and the d/f block; RDKit leaves those without implicit
    hydrogens.
    """
    element = symbol.rstrip("+-").capitalize()
    try:
        atomic_number = _PERIODIC_TABLE.GetAtomicNumber(element)
    except RuntimeError:
        return -1, ()
    default = _PERIODIC_TABLE.GetDefaultValence(atomic_number)
    valences = tuple(
        valence
        for valence in _PERIODIC_TABLE.GetValenceList(atomic_number)
        if valence >= 0
    )
    if default < 0 or not valences:
        return -1, ()
    return default, valences


def _implicit_hydrogens(symbol: str, bond_order_sum: float) -> int:
    """Reproduce RDKit's implicit hydrogen count for one finished atom.

    RDKit does not fill an atom up to its *largest* allowed valence; it takes
    the smallest entry of ``GetValenceList`` that covers the bond order the atom
    actually uses. Sulfur is the case that matters here: a thioether uses 2 of
    ``[2, 4, 6]`` and therefore carries no hydrogen, where a single maximum of 6
    invents four.

    Aromatic atoms are handled the way ``Atom::calcExplicitValence`` does rather
    than by kekulising: an aromatic bond contributes 1.5, so the accumulated
    valence overshoots by 0.5 per bond whenever the Kekule structure gives the
    atom a lone pair instead of a double bond. RDKit falls back to the largest
    allowed valence at or below the accumulated one whenever the gap is at most
    1.5, which is what turns thiophene sulfur (2 x 1.5 = 3.0) into 2 and leaves
    benzene carbon at 3. Deciding it locally, from the 1.5 sums the scanner
    already keeps, needs no ring perception and was checked against RDKit on
    every atom of all 7,551 NPLIB1 train and test targets.
    """
    default, valences = _rdkit_valences(symbol)
    if default < 0:
        return 0
    aromatic = symbol.islower()
    used = bond_order_sum
    if aromatic and used > default:
        allowed = default
        for valence in valences:
            if valence > used:
                break
            allowed = valence
        if allowed + 1.5 >= used:
            used = float(allowed)
    # C++ std::round sends halves away from zero, where Python's round() sends
    # them to even; an aromatic atom outside a ring can still reach x.5 here.
    explicit_valence = floor(used + 0.5)
    if aromatic:
        return max(default - explicit_valence, 0)
    for valence in valences:
        if explicit_valence <= valence:
            return valence - explicit_valence
    return 0


def _terminal_hydrogens(state: "_GrammarState") -> int:
    """Count the hydrogens RDKit will read back from a finished SAFE string.

    A terminal state names every heavy atom and every bond, so its hydrogen
    count is a single number rather than the interval
    :meth:`_GrammarState.hydrogen_bounds` has to report while heavy atoms may
    still arrive.
    """
    total = 0
    for atom, symbol in state.atom_symbols.items():
        if atom in state.bracket_atoms:
            # RDKit sets noImplicit on every bracket atom, so whatever hydrogen
            # count stands between the brackets is the whole count: "[C]" holds
            # none and "[O-]" holds none either. This is why a formal charge
            # never moves the count -- SMILES cannot write a charge outside
            # brackets -- and why "[CH2]" holds two rather than the three a
            # valence model would add.
            total += state.explicit_hydrogens[atom]
            continue
        total += _implicit_hydrogens(symbol, state.bond_order_sums[atom])
    return total


def _partial_bracket_symbol(content: str) -> str | None:
    if not content or (content.isdigit() and len(content) <= 3):
        return ""
    match = _BRACKET_ATOM.fullmatch(content)
    if not match:
        return None
    element = match.group("element")
    isotope = match.group("isotope")
    if isotope and element != "*":
        atomic_number = _PERIODIC_TABLE.GetAtomicNumber(element.capitalize())
        if _PERIODIC_TABLE.GetMassForIsotope(atomic_number, int(isotope)) <= 0:
            return None
    if element == "N" and "+" in content[match.end("element") :]:
        return "N+"
    return element


def _has_mass_viable_bracket_completion(
    text: str,
    target_mass: float,
    valence_slack: float,
    tolerance: float,
) -> bool:
    bracket = text.rfind("[")
    if bracket <= text.rfind("]"):
        return True
    content = text[bracket + 1 :]
    if content and not content.isdigit():
        return True
    base = _scan_base(text)
    if base is None:
        return False
    return any(
        (state := _scan_continuation(base, element + "]")) is not None
        and state.minimum_mass(valence_slack) <= target_mass + tolerance
        for element in _BRACKET_COMPLETION_ELEMENTS
    )


def _has_mass_viable_atom_completion(
    text: str,
    target_mass: float,
    valence_slack: float,
    tolerance: float,
    base: _ScanBase | None = None,
) -> bool:
    if base is None:
        base = _scan_base(text)
        if base is None:
            return False
    return any(
        (state := _scan_continuation(base, atom)) is not None
        and state.minimum_mass(valence_slack) <= target_mass + tolerance
        and _has_reachable_exact_mass(
            state,
            target_mass,
            valence_slack,
            tolerance,
        )
        for atom in _ATOM_COMPLETIONS
    )


def _has_mass_viable_percent_completion(
    text: str,
    target_mass: float,
    valence_slack: float,
    tolerance: float,
) -> bool:
    percent = text.rfind("%")
    suffix = text[percent + 1 :] if percent >= 0 else ""
    if percent < 0 or len(suffix) >= 2 or (suffix and not suffix.isdigit()):
        return True
    base = _scan_base(text[: percent + 1])
    if base is None:
        return False
    completions = (str(value) for value in range(10, 100))
    return any(
        completion.startswith(suffix)
        and (state := _scan_continuation(base, completion)) is not None
        and state.minimum_mass(valence_slack) <= target_mass + tolerance
        for completion in completions
    )


@lru_cache(maxsize=4)
def _reachable_valences(max_mass_bucket: int) -> np.ndarray:
    max_index = max_mass_bucket * _REACHABILITY_SCALE
    unreachable = np.iinfo(np.int16).min
    valences = np.full(max_index + 1, unreachable, dtype=np.int16)
    valences[0] = 0
    for symbol, valence in _ATOM_VALENCES.items():
        weight = round(_ATOM_MASSES[symbol] * _REACHABILITY_SCALE)
        remaining = max_index // weight
        power = 1
        while remaining:
            count = min(power, remaining)
            shift = weight * count
            source = valences[:-shift].copy()
            reachable = source >= 0
            source[reachable] += valence * count
            np.maximum(valences[shift:], source, out=valences[shift:])
            remaining -= count
            power *= 2
    return valences


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
    open_ring_orders: dict[str, float] | None = None
    bond_counts: dict[int, int] | None = None
    bond_limits: dict[int, int] | None = None
    bond_order_sums: dict[int, float] | None = None
    valence_usage_sums: dict[int, float] | None = None
    valence_limits: dict[int, float] | None = None
    atom_masses: dict[int, float] | None = None
    atom_symbols: dict[int, str] | None = None
    explicit_hydrogens: dict[int, int] | None = None
    bracket_atoms: set[int] | None = None
    bonds: set[tuple[int, int]] | None = None
    pending_bond_order: float | None = None
    incomplete_token: bool = False

    def __post_init__(self) -> None:
        if self.open_rings is None:
            self.open_rings = {}
        if self.branch_atoms is None:
            self.branch_atoms = []
        if self.open_ring_orders is None:
            self.open_ring_orders = {}
        if self.bond_counts is None:
            self.bond_counts = {}
        if self.bond_limits is None:
            self.bond_limits = {}
        if self.bond_order_sums is None:
            self.bond_order_sums = {}
        if self.valence_usage_sums is None:
            self.valence_usage_sums = {}
        if self.valence_limits is None:
            self.valence_limits = {}
        if self.atom_masses is None:
            self.atom_masses = {}
        if self.atom_symbols is None:
            self.atom_symbols = {}
        if self.explicit_hydrogens is None:
            self.explicit_hydrogens = {}
        if self.bracket_atoms is None:
            self.bracket_atoms = set()
        if self.bonds is None:
            self.bonds = set()

    def copy(self) -> "_GrammarState":
        """Clone the state so a candidate token can be scanned on top of it.

        Every mutable container is duplicated, so advancing the clone leaves the
        state it was taken from untouched. ``__new__`` skips ``__post_init__``
        because every field is assigned here; a field added without a line here
        raises ``AttributeError`` on first use instead of leaking a shared
        container.
        """
        clone = _GrammarState.__new__(_GrammarState)
        clone.expect_atom = self.expect_atom
        clone.allow_bond = self.allow_bond
        clone.allow_ring = self.allow_ring
        clone.branch_depth = self.branch_depth
        clone.atom_index = self.atom_index
        clone.current_atom = self.current_atom
        clone.branch_atoms = list(self.branch_atoms)
        clone.open_rings = dict(self.open_rings)
        clone.open_ring_orders = dict(self.open_ring_orders)
        clone.bond_counts = dict(self.bond_counts)
        clone.bond_limits = dict(self.bond_limits)
        clone.bond_order_sums = dict(self.bond_order_sums)
        clone.valence_usage_sums = dict(self.valence_usage_sums)
        clone.valence_limits = dict(self.valence_limits)
        clone.atom_masses = dict(self.atom_masses)
        clone.atom_symbols = dict(self.atom_symbols)
        clone.explicit_hydrogens = dict(self.explicit_hydrogens)
        clone.bracket_atoms = set(self.bracket_atoms)
        clone.bonds = set(self.bonds)
        clone.pending_bond_order = self.pending_bond_order
        clone.incomplete_token = self.incomplete_token
        return clone

    @property
    def terminal(self) -> bool:
        return (
            not self.expect_atom
            and self.branch_depth == 0
            and not self.open_rings
            and not self.incomplete_token
        )

    def minimum_mass(self, valence_slack: float) -> float:
        active_atoms = set(self.branch_atoms)
        active_atoms.update(self.open_rings.values())
        if self.current_atom is not None:
            active_atoms.add(self.current_atom)
        implicit_hydrogens = 0.0
        for atom, valence in self.valence_limits.items():
            if atom not in active_atoms:
                implicit_hydrogens += max(
                    valence
                    - self.bond_order_sums[atom]
                    - self.explicit_hydrogens[atom],
                    0.0,
                )
        implicit_hydrogens = max(implicit_hydrogens - valence_slack, 0.0)
        explicit_hydrogens = sum(self.explicit_hydrogens.values())
        return (
            sum(self.atom_masses.values())
            + (implicit_hydrogens + explicit_hydrogens) * _HYDROGEN_MASS
        )

    def hydrogen_bounds(self, valence_slack: float) -> tuple[int, int]:
        active_atoms = set(self.branch_atoms)
        active_atoms.update(self.open_rings.values())
        if self.current_atom is not None:
            active_atoms.add(self.current_atom)
        available = {
            atom: max(
                valence - self.bond_order_sums[atom] - self.explicit_hydrogens[atom],
                0.0,
            )
            for atom, valence in self.valence_limits.items()
        }
        explicit = sum(self.explicit_hydrogens.values())
        sealed = sum(
            hydrogens
            for atom, hydrogens in available.items()
            if atom not in active_atoms
        )
        minimum = explicit + max(sealed - valence_slack, 0.0)
        maximum = explicit + sum(available.values())
        return max(floor(minimum + 1e-6), 0), max(ceil(maximum - 1e-6), 0)


def _has_reachable_exact_mass(
    state: _GrammarState,
    target_mass: float,
    valence_slack: float,
    tolerance: float,
) -> bool:
    heavy_mass = sum(state.atom_masses.values())
    residual = target_mass - heavy_mass
    if residual < -tolerance:
        return False
    minimum_hydrogens, maximum_existing_hydrogens = state.hydrogen_bounds(valence_slack)
    maximum_total_hydrogens = max(floor((residual + tolerance) / _HYDROGEN_MASS), 0)
    if minimum_hydrogens > maximum_total_hydrogens:
        return False
    bucket = max(ceil(target_mass / 100.0) * 100, 100)
    reachable_valences = _reachable_valences(bucket)
    tolerance_bins = ceil(tolerance * _REACHABILITY_SCALE) + 3
    for hydrogens in range(minimum_hydrogens, maximum_total_hydrogens + 1):
        future_mass = residual - hydrogens * _HYDROGEN_MASS
        center = round(future_mass * _REACHABILITY_SCALE)
        lower = max(center - tolerance_bins, 0)
        upper = min(center + tolerance_bins + 1, len(reachable_valences))
        required_future_valence = max(
            hydrogens - maximum_existing_hydrogens,
            0,
        )
        if lower < upper and np.any(
            reachable_valences[lower:upper] >= required_future_valence
        ):
            return True
    return False


def _advance(
    state: _GrammarState,
    text: str,
    *,
    stop_before_lookahead: bool = False,
) -> int | None:
    """Fold ``text`` into ``state`` and report how much of it was consumed.

    Returns ``None`` when ``text`` cannot continue the state, otherwise the
    index one past the last consumed character. Every decision reads the state,
    the current character and characters *after* it, never characters before
    it, so a scan may be resumed from any index this returns.

    With ``stop_before_lookahead`` the scan stops in front of a trailing element
    whose parse depends on characters beyond ``text``: an unterminated "[", a
    "%" that has fewer than two characters behind it, or a final "B"/"C" that a
    following "r"/"l" would turn into a two-character atom. Splitting there lets
    a caller reuse the state for many different continuations.
    """

    def within_valence(atom: int) -> bool:
        return (
            state.valence_usage_sums[atom] + state.explicit_hydrogens[atom]
            <= state.valence_limits[atom] + 1e-6
        )

    def add_atom(
        symbol: str,
        explicit_hydrogens: int = 0,
        *,
        bracketed: bool = False,
    ) -> bool:
        previous_atom = state.current_atom
        state.atom_index += 1
        state.current_atom = state.atom_index
        if bracketed:
            state.bracket_atoms.add(state.atom_index)
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
        state.valence_limits[state.atom_index] = {
            "H": 1.0,
            "F": 1.0,
            "Cl": 1.0,
            "Br": 1.0,
            "I": 1.0,
            "B": 3.0,
            "b": 3.0,
            "C": 4.0,
            "c": 4.0,
            "N": 3.0,
            "N+": 4.0,
            "n": 3.0,
            "O": 2.0,
            "o": 2.0,
            "P": 5.0,
            "p": 3.0,
            "S": 6.0,
            "s": 2.0,
        }.get(symbol, 4.0)
        state.bond_counts[state.atom_index] = 0
        state.bond_order_sums[state.atom_index] = 0.0
        state.valence_usage_sums[state.atom_index] = 0.0
        mass_symbol = symbol.rstrip("+")
        state.atom_masses[state.atom_index] = _ATOM_MASSES.get(
            mass_symbol.capitalize() if len(mass_symbol) == 1 else mass_symbol,
            0.0,
        )
        state.atom_symbols[state.atom_index] = symbol
        state.explicit_hydrogens[state.atom_index] = explicit_hydrogens
        if previous_atom is not None:
            previous_symbol = state.atom_symbols[previous_atom]
            aromatic_bond = previous_symbol in "bcnops" and symbol in "bcnops"
            bond_order = state.pending_bond_order or (1.5 if aromatic_bond else 1.0)
            # A chain or branch bond always reaches a brand new atom index, so it
            # cannot duplicate one; only a ring closure can, which is checked there.
            state.bonds.add((previous_atom, state.atom_index))
            state.bond_counts[previous_atom] += 1
            state.bond_counts[state.atom_index] += 1
            state.bond_order_sums[previous_atom] += bond_order
            state.bond_order_sums[state.atom_index] += bond_order
            valence_usage = 1.0 if aromatic_bond else bond_order
            state.valence_usage_sums[previous_atom] += valence_usage
            state.valence_usage_sums[state.atom_index] += valence_usage
            if (
                state.bond_counts[previous_atom] > state.bond_limits[previous_atom]
                or not within_valence(previous_atom)
                or not within_valence(state.atom_index)
            ):
                return False
        elif not within_valence(state.atom_index):
            return False
        state.pending_bond_order = None
        state.expect_atom = False
        state.allow_bond = False
        state.allow_ring = False
        return True

    index = 0
    while index < len(text):
        char = text[index]
        if (
            stop_before_lookahead
            and index + 1 == len(text)
            and char in _TWO_CHARACTER_ATOM_STARTS
        ):
            return index
        if char == "[":
            close = text.find("]", index + 1)
            if close < 0:
                if stop_before_lookahead:
                    return index
                symbol = _partial_bracket_symbol(text[index + 1 :])
                if symbol is None:
                    return None
                if symbol and not add_atom(symbol, bracketed=True):
                    return None
                state.incomplete_token = True
                return len(text)
            symbol = _partial_bracket_symbol(text[index + 1 : close])
            if not symbol:
                return None
            content = text[index + 1 : close]
            hydrogen_match = re.search(r"H(\d*)", content)
            explicit_hydrogens = (
                int(hydrogen_match.group(1) or "1") if hydrogen_match else 0
            )
            if not add_atom(symbol, explicit_hydrogens, bracketed=True):
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
            state.pending_bond_order = _BOND_ORDERS[char]
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
            state.pending_bond_order = None
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
            state.pending_bond_order = None
            index += 1
            continue
        if char == ".":
            if state.expect_atom:
                return None
            state.current_atom = None
            state.expect_atom = True
            state.allow_bond = False
            state.allow_ring = False
            state.pending_bond_order = None
            index += 1
            continue
        if char == "%":
            remaining = text[index + 1 :]
            if len(remaining) < 2:
                if stop_before_lookahead:
                    return index
                if remaining.isdigit() or not remaining:
                    state.incomplete_token = True
                    return len(text)
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
            explicit_order = state.pending_bond_order or 0.0
            state.open_ring_orders[label] = explicit_order
            atom = state.current_atom
            bond_order = explicit_order or 1.0
            valence_usage = bond_order
        elif opening_atom == state.current_atom:
            return None
        elif (
            min(opening_atom, state.current_atom),
            max(opening_atom, state.current_atom),
        ) in state.bonds:
            # RDKit refuses a ring closure that duplicates an existing bond
            # ("C12CC12", "C%99C%99", "c1ccccc1.C12.C12"), and the grammar used to
            # accept all three as finished molecules: 33 of the 91 unparsable
            # terminal strings of the 803-spectrum run are exactly this.
            return None
        else:
            del state.open_rings[label]
            state.bonds.add(
                (
                    min(opening_atom, state.current_atom),
                    max(opening_atom, state.current_atom),
                )
            )
            explicit_order = state.open_ring_orders.pop(label)
            atom = state.current_atom
            aromatic_bond = (
                state.atom_symbols[opening_atom] in "bcnops"
                and state.atom_symbols[atom] in "bcnops"
            )
            bond_order = (
                state.pending_bond_order
                or explicit_order
                or (1.5 if aromatic_bond else 1.0)
            )
            reserved_order = explicit_order or 1.0
            state.bond_order_sums[opening_atom] += bond_order - reserved_order
            reserved_usage = explicit_order or 1.0
            valence_usage = 1.0 if aromatic_bond else bond_order
            state.valence_usage_sums[opening_atom] += valence_usage - reserved_usage
        state.bond_counts[atom] += 1
        state.bond_order_sums[atom] += bond_order
        state.valence_usage_sums[atom] += valence_usage
        if (
            state.bond_counts[atom] > state.bond_limits[atom]
            or not within_valence(atom)
            or (opening_atom is not None and not within_valence(opening_atom))
        ):
            return None
        state.pending_bond_order = None
    return len(text)


def _scan(text: str) -> _GrammarState | None:
    """Parse a SAFE prefix into its grammar state.

    Deliberately uncached. Memoising this was measured at 0.98x on a realistic
    prefix walk while retaining hundreds of megabytes of tracked containers and
    turning ~45 garbage collections into ~2,500; the incremental path is what
    removes the repeated work. All mutation happens in :func:`_advance`.
    """
    state = _GrammarState()
    if _advance(state, text) is None:
        return None
    return state


class _ScanBase(NamedTuple):
    """A scanned prefix plus the trailing characters left for a continuation.

    ``state`` covers everything before ``pending``; ``pending`` holds the few
    characters :func:`_advance` refused to consume because their parse depends
    on what comes next. Callers must not mutate ``state``.
    """

    state: _GrammarState
    pending: str


def _scan_base(text: str) -> _ScanBase | None:
    """Scan the part of ``text`` that no continuation can reinterpret.

    Returns ``None`` only when ``text`` is already invalid, in which case
    ``_scan(text + suffix)`` is ``None`` for every ``suffix``.
    """
    state = _GrammarState()
    consumed = _advance(state, text, stop_before_lookahead=True)
    if consumed is None:
        return None
    return _ScanBase(state, text[consumed:])


def _scan_continuation(base: _ScanBase, suffix: str) -> _GrammarState | None:
    """Return the state of ``base`` extended by ``suffix``.

    Equivalent to ``_scan(text + suffix)`` for the ``text`` that produced
    ``base``, but the shared prefix is scanned once instead of once per
    candidate suffix.
    """
    state = base.state.copy()
    if _advance(state, base.pending + suffix) is None:
        return None
    return state


def _extend_base(base: _ScanBase, suffix: str) -> _ScanBase | None:
    """Return the base of ``text + suffix`` for the ``text`` behind ``base``."""
    state = base.state.copy()
    text = base.pending + suffix
    consumed = _advance(state, text, stop_before_lookahead=True)
    if consumed is None:
        return None
    return _ScanBase(state, text[consumed:])


def _has_hydrogen_only_exact_mass(
    state: _GrammarState,
    target_mass: float,
    valence_slack: float,
    tolerance: float,
) -> bool:
    """Report whether hydrogens alone can put ``state`` on ``target_mass``.

    A terminal state has no freedom left: every heavy atom and every bond is
    named, so its hydrogen count is the single number RDKit will read back and
    the gate is an equality on one mass. A partial state may still grow heavy
    atoms that change how many hydrogens the atoms already placed will keep, so
    there the interval from :meth:`_GrammarState.hydrogen_bounds` is what the
    gate can honestly assert.
    """
    heavy_mass = sum(state.atom_masses.values())
    if state.terminal:
        hydrogens = _terminal_hydrogens(state)
        return abs(heavy_mass + hydrogens * _HYDROGEN_MASS - target_mass) <= tolerance
    minimum_hydrogens, maximum_hydrogens = state.hydrogen_bounds(valence_slack)
    return any(
        abs(heavy_mass + hydrogens * _HYDROGEN_MASS - target_mass) <= tolerance
        for hydrogens in range(minimum_hydrogens, maximum_hydrogens + 1)
    )


def _has_structurally_viable_continuation(
    text: str,
    state: _GrammarState,
    target_mass: float,
    valence_slack: float,
    tolerance: float,
    base: _ScanBase | None = None,
) -> bool:
    if state.incomplete_token:
        return True
    if state.terminal and _has_hydrogen_only_exact_mass(
        state,
        target_mass,
        valence_slack,
        tolerance,
    ):
        return True
    if base is None:
        base = _scan_base(text)
        if base is None:
            return False
    if state.expect_atom and state.allow_ring:
        for label in state.open_rings:
            closed = _scan_continuation(base, label)
            if closed is not None and _has_reachable_exact_mass(
                closed,
                target_mass,
                valence_slack,
                tolerance,
            ):
                return True
    if state.expect_atom:
        return _has_mass_viable_atom_completion(
            text,
            target_mass,
            valence_slack,
            tolerance,
            base,
        )
    if state.current_atom is not None and (
        state.bond_counts[state.current_atom] < state.bond_limits[state.current_atom]
    ):
        if _has_mass_viable_atom_completion(
            text,
            target_mass,
            valence_slack,
            tolerance,
            base,
        ):
            return True
    for label in state.open_rings:
        closed = _scan_continuation(base, label)
        if closed is not None and _has_reachable_exact_mass(
            closed,
            target_mass,
            valence_slack,
            tolerance,
        ):
            if _has_structurally_viable_continuation(
                text + label,
                closed,
                target_mass,
                valence_slack,
                tolerance,
                _extend_base(base, label),
            ):
                return True
    if state.branch_depth:
        closed = _scan_continuation(base, ")")
        if closed is not None and _has_reachable_exact_mass(
            closed,
            target_mass,
            valence_slack,
            tolerance,
        ):
            return _has_structurally_viable_continuation(
                text + ")",
                closed,
                target_mass,
                valence_slack,
                tolerance,
                _extend_base(base, ")"),
            )
    disconnected = _scan_continuation(base, ".")
    return disconnected is not None and _has_mass_viable_atom_completion(
        text + ".",
        target_mass,
        valence_slack,
        tolerance,
        _extend_base(base, "."),
    )


def _has_vocabulary_completion(
    text: str,
    token_strings: Sequence[str],
    target_mass: float,
    valence_slack: float,
    tolerance: float,
) -> bool:
    has_open_bracket = text.rfind("[") > text.rfind("]")
    trailing_percent = text.rfind("%") > max(text.rfind("["), text.rfind("]"))
    base = _scan_base(text)
    if base is None:
        return False
    for token in token_strings:
        if has_open_bracket:
            close = token.find("]")
            nested_open = token.find("[")
            if close < 0 or (nested_open >= 0 and nested_open < close):
                continue
        elif trailing_percent and (not token or not token[0].isdigit()):
            continue
        completed = _scan_continuation(base, token)
        if completed is None or completed.incomplete_token:
            continue
        if not _has_reachable_exact_mass(
            completed,
            target_mass,
            valence_slack,
            tolerance,
        ):
            continue
        if _has_structurally_viable_continuation(
            text + token,
            completed,
            target_mass,
            valence_slack,
            tolerance,
            _extend_base(base, token),
        ):
            return True
    return False


class SafeGrammarMask:
    """Retain the full lexical SAFE support for a contiguous committed prefix."""

    def __init__(
        self,
        token_strings: Sequence[str],
        decode_prefix: Callable[[Sequence[int]], str],
        *,
        eos_token_id: int,
        mask_token_id: int | None = None,
        special_token_ids: Sequence[int],
        forbidden_token_ids: Sequence[int] = (),
        ppm_tolerance: float = 10.0,
        valence_slack: float = 4.0,
        mass_reachability_prune: bool = False,
    ) -> None:
        self.token_strings = tuple(token_strings)
        self.decode_prefix = decode_prefix
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id
        self.special_token_ids = frozenset(special_token_ids)
        # Withholding support here rather than only zeroing the sampler logits
        # keeps these tokens out of the per-token mass reachability search,
        # which is the dominant cost of a constrained decode.
        self.forbidden_token_ids = frozenset(forbidden_token_ids)
        self._blocked_token_ids = self.special_token_ids | self.forbidden_token_ids
        self.ppm_tolerance = ppm_tolerance
        self.valence_slack = valence_slack
        # Off by default: the paper's syntax mask carries no chemical mass
        # reachability, so enabling this is a documented deviation.
        self.mass_reachability_prune = mass_reachability_prune

    @lru_cache(maxsize=32_768)
    def _has_syntactic_completion(self, text: str) -> bool:
        """Report whether any vocabulary token can close a partial SAFE token.

        A partial token such as the isotope digits opened by "[" stays lexically
        valid on its own, so admitting it without checking that some token can
        finish it lets the decoder walk into a state with no legal successor.
        """
        has_open_bracket = text.rfind("[") > text.rfind("]")
        trailing_percent = text.rfind("%") > max(text.rfind("["), text.rfind("]"))
        base = _scan_base(text)
        if base is None:
            return False
        for token in self.token_strings:
            if has_open_bracket:
                close = token.find("]")
                nested_open = token.find("[")
                if close < 0 or (nested_open >= 0 and nested_open < close):
                    continue
            elif trailing_percent and (not token or not token[0].isdigit()):
                continue
            completed = _scan_continuation(base, token)
            if completed is not None and not completed.incomplete_token:
                return True
        return False

    def _token_is_valid(
        self,
        prefix: str,
        state: _GrammarState,
        base: _ScanBase,
        token_id: int,
    ) -> bool:
        """Report whether ``token_id`` is in the lexical support of ``prefix``."""
        if token_id == self.eos_token_id:
            return state.terminal
        if token_id in self._blocked_token_ids:
            return False
        token = self.token_strings[token_id]
        completed = _scan_continuation(base, token)
        if completed is None:
            return False
        if completed.incomplete_token:
            # Lexical SAFE support excludes partial tokens the vocabulary cannot
            # finish; admitting them creates unreachable dead ends.
            return self._has_syntactic_completion(prefix + token)
        return True

    @lru_cache(maxsize=32_768)
    def _valid_token_ids(self, prefix: str) -> tuple[int, ...]:
        state = _scan(prefix)
        if state is None:
            return ()
        # A prefix that scans always has a base; only its deferred tail can fail.
        base = _scan_base(prefix)
        assert base is not None
        return tuple(
            token_id
            for token_id in range(len(self.token_strings))
            if self._token_is_valid(prefix, state, base, token_id)
        )

    def _token_is_mass_reachable(
        self,
        prefix: str,
        state: _GrammarState,
        base: _ScanBase,
        token_id: int,
        target_mass: float,
        tolerance: float,
    ) -> bool:
        """Report whether ``token_id`` keeps ``target_mass`` reachable."""
        if token_id == self.eos_token_id:
            return state.terminal and _has_hydrogen_only_exact_mass(
                state, target_mass, self.valence_slack, tolerance
            )
        if token_id in self._blocked_token_ids:
            return False
        token = self.token_strings[token_id]
        text = prefix + token
        token_base = _extend_base(base, token)
        completed = None if token_base is None else _scan_continuation(token_base, "")
        if completed is None:
            return False
        if completed.incomplete_token:
            return _has_vocabulary_completion(
                text,
                self.token_strings,
                target_mass,
                self.valence_slack,
                tolerance,
            )
        return _has_reachable_exact_mass(
            completed, target_mass, self.valence_slack, tolerance
        ) and _has_structurally_viable_continuation(
            text,
            completed,
            target_mass,
            self.valence_slack,
            tolerance,
            token_base,
        )

    @lru_cache(maxsize=32_768)
    def _mass_reachable_token_ids(
        self, prefix: str, target_mass: float
    ) -> tuple[int, ...]:
        state = _scan(prefix)
        if state is None:
            return ()
        # A prefix that scans always has a base; only its deferred tail can fail.
        base = _scan_base(prefix)
        assert base is not None
        tolerance = self.ppm_tolerance * 1e-6 * target_mass
        return tuple(
            token_id
            for token_id in range(len(self.token_strings))
            if self._token_is_mass_reachable(
                prefix, state, base, token_id, target_mass, tolerance
            )
        )

    def admits(
        self,
        prefix_ids: Sequence[int],
        token_id: int,
        target_mass: float | None = None,
    ) -> bool:
        """Report whether one token is in the support ``__call__`` would leave.

        Asking about a single token instead of building the whole support is what
        makes a token-by-token walk of every gold answer affordable, which is the
        acceptance gate for any change to this mask
        (``scripts/audit_marlin_gold_mask_walk.py``).
        """
        if self.mask_token_id is not None and self.mask_token_id in prefix_ids:
            return False
        prefix = self.decode_prefix(prefix_ids)
        state = _scan(prefix)
        if state is None:
            return False
        base = _scan_base(prefix)
        assert base is not None
        if self.mass_reachability_prune and target_mass is not None:
            return self._token_is_mass_reachable(
                prefix,
                state,
                base,
                token_id,
                target_mass,
                self.ppm_tolerance * 1e-6 * target_mass,
            )
        return self._token_is_valid(prefix, state, base, token_id)

    def __call__(
        self,
        prefix_ids: Sequence[int],
        logits: torch.Tensor,
        target_mass: float | None = None,
    ) -> torch.Tensor:
        # Grammar state is defined only for a contiguous committed prefix.
        # Give positions after an unresolved hole no support so confidence-order
        # sampling cannot commit a suffix that invalidates the eventual prefix.
        if self.mask_token_id is not None and self.mask_token_id in prefix_ids:
            return torch.full_like(logits, -torch.inf)
        prefix = self.decode_prefix(prefix_ids)
        if self.mass_reachability_prune and target_mass is not None:
            valid_ids = self._mass_reachable_token_ids(prefix, target_mass)
        else:
            valid_ids = self._valid_token_ids(prefix)
        if not valid_ids:
            return torch.full_like(logits, -torch.inf)
        # Scattering ~1,300 python ints into the tensor costs about 250 us per
        # call; selecting through a cached boolean mask produces the identical
        # tensor in about 7 us, and this runs once per candidate per position.
        return torch.where(
            self._support_mask(valid_ids, logits.device),
            logits,
            torch.full_like(logits, -torch.inf),
        )

    @lru_cache(maxsize=32_768)
    def _support_mask(
        self, valid_ids: tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        mask = torch.zeros(len(self.token_strings), dtype=torch.bool, device=device)
        mask[list(valid_ids)] = True
        return mask
