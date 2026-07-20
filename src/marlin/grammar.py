"""Conservative lexical grammar mask for incremental SAFE/SMILES decoding."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from math import ceil, floor
from typing import Callable, Sequence

import numpy as np
import torch
from rdkit import Chem


_TWO_CHARACTER_ATOMS = ("Br", "Cl")
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
    return any(
        (state := _scan(text + element + "]")) is not None
        and state.minimum_mass(valence_slack) <= target_mass + tolerance
        for element in _BRACKET_COMPLETION_ELEMENTS
    )


def _has_mass_viable_atom_completion(
    text: str,
    target_mass: float,
    valence_slack: float,
    tolerance: float,
) -> bool:
    return any(
        (state := _scan(text + atom)) is not None
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
    completions = (str(value) for value in range(10, 100))
    return any(
        completion.startswith(suffix)
        and (state := _scan(text[: percent + 1] + completion)) is not None
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


def _scan(text: str) -> _GrammarState | None:
    state = _GrammarState()

    def within_valence(atom: int) -> bool:
        return (
            state.valence_usage_sums[atom] + state.explicit_hydrogens[atom]
            <= state.valence_limits[atom] + 1e-6
        )

    def add_atom(symbol: str, explicit_hydrogens: int = 0) -> bool:
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
            content = text[index + 1 : close]
            hydrogen_match = re.search(r"H(\d*)", content)
            explicit_hydrogens = (
                int(hydrogen_match.group(1) or "1") if hydrogen_match else 0
            )
            if not add_atom(symbol, explicit_hydrogens):
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
            explicit_order = state.pending_bond_order or 0.0
            state.open_ring_orders[label] = explicit_order
            atom = state.current_atom
            bond_order = explicit_order or 1.0
            valence_usage = bond_order
        elif opening_atom == state.current_atom:
            return None
        else:
            del state.open_rings[label]
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
    return state


def _has_hydrogen_only_exact_mass(
    state: _GrammarState,
    target_mass: float,
    valence_slack: float,
    tolerance: float,
) -> bool:
    heavy_mass = sum(state.atom_masses.values())
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
    if state.expect_atom and state.allow_ring:
        for label in state.open_rings:
            closed = _scan(text + label)
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
        )
    if state.current_atom is not None and (
        state.bond_counts[state.current_atom] < state.bond_limits[state.current_atom]
    ):
        if _has_mass_viable_atom_completion(
            text,
            target_mass,
            valence_slack,
            tolerance,
        ):
            return True
    for label in state.open_rings:
        closed = _scan(text + label)
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
            ):
                return True
    if state.branch_depth:
        closed = _scan(text + ")")
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
            )
    disconnected = _scan(text + ".")
    return disconnected is not None and _has_mass_viable_atom_completion(
        text + ".",
        target_mass,
        valence_slack,
        tolerance,
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
    for token in token_strings:
        if has_open_bracket:
            close = token.find("]")
            nested_open = token.find("[")
            if close < 0 or (nested_open >= 0 and nested_open < close):
                continue
        elif trailing_percent and (not token or not token[0].isdigit()):
            continue
        completed = _scan(text + token)
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
        ppm_tolerance: float = 10.0,
        valence_slack: float = 4.0,
    ) -> None:
        self.token_strings = tuple(token_strings)
        self.decode_prefix = decode_prefix
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id
        self.special_token_ids = frozenset(special_token_ids)
        self.ppm_tolerance = ppm_tolerance
        self.valence_slack = valence_slack

    @lru_cache(maxsize=32_768)
    def _valid_token_ids(self, prefix: str) -> tuple[int, ...]:
        state = _scan(prefix)
        if state is None:
            return ()
        valid_ids = []
        for token_id, token in enumerate(self.token_strings):
            if token_id == self.eos_token_id:
                valid = state.terminal
            elif token_id in self.special_token_ids:
                valid = False
            else:
                valid = _scan(prefix + token) is not None
            if valid:
                valid_ids.append(token_id)
        return tuple(valid_ids)

    def __call__(
        self,
        prefix_ids: Sequence[int],
        logits: torch.Tensor,
        target_mass: float | None = None,
    ) -> torch.Tensor:
        # The paper does not specify grammar state for holes inside a partially
        # revealed block. Conservatively avoid false pruning until every earlier
        # position is known; the completed block still passes strict SAFE decode.
        if self.mask_token_id is not None and self.mask_token_id in prefix_ids:
            return logits.clone()
        prefix = self.decode_prefix(prefix_ids)
        constrained = torch.full_like(logits, -torch.inf)
        valid_ids = self._valid_token_ids(prefix)
        if valid_ids:
            indices = list(valid_ids)
            constrained[indices] = logits[indices]
        return constrained
