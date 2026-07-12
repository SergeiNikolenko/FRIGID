"""Deterministic, formula-constrained STONED-SELFIES candidate expansion."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
import random
from typing import Iterable, Sequence

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
import selfies as sf


SUPPORTED_OPERATIONS = ("replacement", "insertion", "deletion", "paired_swap")
RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class StonedCandidate:
    smiles: str
    inchi_key_connectivity: str
    raw_selfies: str
    mutation_depth: int
    mutation_operations: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class StonedProposal:
    proposal_index: int
    mutation_depth: int
    mutation_operations: tuple[dict[str, object], ...]
    raw_selfies: str
    resulting_smiles: str
    inchi_key_connectivity: str
    accepted: bool
    rejection_reason: str

    def as_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["mutation_operations"] = json.dumps(
            self.mutation_operations, sort_keys=True, separators=(",", ":")
        )
        return row


@dataclass
class StonedStatistics:
    proposals_considered: int = 0
    accepted_unique: int = 0
    valid_molecules: int = 0
    exact_formula_molecules: int = 0
    rejection_counts: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, object]:
        considered = max(self.proposals_considered, 1)
        valid = max(self.valid_molecules, 1)
        return {
            "proposals_considered": self.proposals_considered,
            "accepted_unique": self.accepted_unique,
            "valid_molecules": self.valid_molecules,
            "exact_formula_molecules": self.exact_formula_molecules,
            "validity_rate": self.valid_molecules / considered,
            "exact_formula_survival_rate": self.exact_formula_molecules / considered,
            "exact_formula_given_valid_rate": self.exact_formula_molecules / valid,
            "rejection_counts": dict(sorted(self.rejection_counts.items())),
        }


def stable_seed(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def connectivity_key(mol_or_smiles: Chem.Mol | str) -> str | None:
    mol = (
        Chem.MolFromSmiles(mol_or_smiles)
        if isinstance(mol_or_smiles, str)
        else mol_or_smiles
    )
    if mol is None:
        return None
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    return key.split("-", maxsplit=1)[0] if key else None


def canonicalize_smiles(smiles: str) -> tuple[Chem.Mol, str, str, str] | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    Chem.RemoveStereochemistry(mol)
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    key = connectivity_key(mol)
    if not key:
        return None
    formula = rdMolDescriptors.CalcMolFormula(mol)
    return mol, canonical, key, formula


def _validate_settings(
    raw_proposals: int,
    max_accepted: int,
    mutation_depths: Sequence[int],
    operations: Sequence[str],
) -> None:
    if raw_proposals <= 0:
        raise ValueError("raw_proposals must be positive")
    if max_accepted <= 0:
        raise ValueError("max_accepted must be positive")
    if not mutation_depths or any(depth <= 0 for depth in mutation_depths):
        raise ValueError("mutation_depths must contain positive integers")
    unknown = set(operations).difference(SUPPORTED_OPERATIONS)
    if not operations or unknown:
        raise ValueError(f"Unsupported mutation operations: {sorted(unknown)}")


def _semantic_alphabet(alphabet: Iterable[str] | None) -> tuple[str, ...]:
    tokens = set(alphabet if alphabet is not None else sf.get_semantic_robust_alphabet())
    tokens.discard("[nop]")
    if not tokens:
        raise ValueError("SELFIES mutation alphabet is empty")
    return tuple(sorted(tokens))


def _mutate_once(
    tokens: list[str],
    operation: str,
    rng: random.Random,
    alphabet: Sequence[str],
) -> dict[str, object]:
    if operation == "replacement":
        position = rng.randrange(len(tokens))
        old = tokens[position]
        choices = [token for token in alphabet if token != old]
        new = rng.choice(choices)
        tokens[position] = new
        return {"operation": operation, "position": position, "old": old, "new": new}
    if operation == "insertion":
        position = rng.randrange(len(tokens) + 1)
        new = rng.choice(alphabet)
        tokens.insert(position, new)
        return {"operation": operation, "position": position, "new": new}
    if operation == "deletion":
        if len(tokens) <= 1:
            return {"operation": operation, "position": -1, "skipped": "minimum_length"}
        position = rng.randrange(len(tokens))
        old = tokens.pop(position)
        return {"operation": operation, "position": position, "old": old}
    if operation == "paired_swap":
        if len(tokens) <= 1:
            return {"operation": operation, "positions": [], "skipped": "minimum_length"}
        first, second = sorted(rng.sample(range(len(tokens)), 2))
        old = [tokens[first], tokens[second]]
        tokens[first], tokens[second] = tokens[second], tokens[first]
        return {
            "operation": operation,
            "positions": [first, second],
            "old": old,
            "new": [tokens[first], tokens[second]],
        }
    raise AssertionError(operation)


def generate_stoned_candidates(
    seed_smiles: str,
    *,
    query_formula: str,
    seed: int,
    raw_proposals: int,
    mutation_depths: Sequence[int] = (1, 2),
    operations: Sequence[str] = ("replacement",),
    max_accepted: int = 64,
    exclude_connectivity_keys: Iterable[str] = (),
    alphabet: Iterable[str] | None = None,
) -> tuple[list[StonedCandidate], list[StonedProposal], StonedStatistics]:
    """Generate a bounded target-blind set of formula-valid SELFIES neighbors."""

    _validate_settings(raw_proposals, max_accepted, mutation_depths, operations)
    prepared = canonicalize_smiles(seed_smiles)
    if prepared is None:
        raise ValueError(f"Invalid or disconnected seed SMILES: {seed_smiles}")
    _, canonical_seed, seed_key, seed_formula = prepared
    if seed_formula != query_formula:
        raise ValueError(
            f"Seed formula mismatch: {seed_formula} != query formula {query_formula}"
        )
    try:
        seed_selfies = sf.encoder(canonical_seed)
    except Exception as exc:
        raise ValueError(f"SELFIES encoding failed for seed {seed_smiles}") from exc
    seed_tokens = list(sf.split_selfies(seed_selfies))
    if not seed_tokens:
        raise ValueError("SELFIES encoding produced no tokens")

    mutation_alphabet = _semantic_alphabet(alphabet)
    rng = random.Random(seed)
    depths = tuple(int(value) for value in mutation_depths)
    operation_names = tuple(operations)
    excluded = set(exclude_connectivity_keys)
    excluded.add(seed_key)
    seen_proposals: set[str] = set()
    accepted: list[StonedCandidate] = []
    proposals: list[StonedProposal] = []
    stats = StonedStatistics()

    for proposal_index in range(raw_proposals):
        depth = depths[proposal_index % len(depths)]
        tokens = list(seed_tokens)
        mutation_trace: list[dict[str, object]] = []
        for _ in range(depth):
            operation = rng.choice(operation_names)
            mutation_trace.append(
                _mutate_once(tokens, operation, rng, mutation_alphabet)
            )
        raw_selfies = "".join(tokens)
        stats.proposals_considered += 1
        rejection_reason = ""
        resulting_smiles = ""
        result_key = ""

        if raw_selfies in seen_proposals:
            rejection_reason = "duplicate_raw_selfies"
        else:
            seen_proposals.add(raw_selfies)
            try:
                decoded = sf.decoder(raw_selfies)
            except Exception:
                decoded = ""
                rejection_reason = "selfies_decode_failure"
            if not rejection_reason:
                mol = Chem.MolFromSmiles(decoded)
                if mol is None:
                    rejection_reason = "rdkit_invalid"
                else:
                    try:
                        Chem.SanitizeMol(mol)
                    except Exception:
                        rejection_reason = "sanitize_failure"
                    if not rejection_reason and len(Chem.GetMolFrags(mol)) != 1:
                        rejection_reason = "disconnected"
                    if not rejection_reason:
                        stats.valid_molecules += 1
                        candidate_formula = rdMolDescriptors.CalcMolFormula(mol)
                        if candidate_formula != query_formula:
                            rejection_reason = "formula_mismatch"
                        else:
                            stats.exact_formula_molecules += 1
                            Chem.RemoveStereochemistry(mol)
                            resulting_smiles = Chem.MolToSmiles(
                                mol, canonical=True, isomericSmiles=False
                            )
                            result_key = connectivity_key(mol) or ""
                            if not result_key:
                                rejection_reason = "inchikey_failure"
                            elif result_key in excluded:
                                rejection_reason = "original_or_duplicate_connectivity"
                            elif len(accepted) >= max_accepted:
                                rejection_reason = "accepted_budget_exceeded"

        was_accepted = not rejection_reason
        if was_accepted:
            excluded.add(result_key)
            accepted.append(
                StonedCandidate(
                    smiles=resulting_smiles,
                    inchi_key_connectivity=result_key,
                    raw_selfies=raw_selfies,
                    mutation_depth=depth,
                    mutation_operations=tuple(mutation_trace),
                )
            )
        else:
            stats.rejection_counts[rejection_reason] += 1
        proposals.append(
            StonedProposal(
                proposal_index=proposal_index,
                mutation_depth=depth,
                mutation_operations=tuple(mutation_trace),
                raw_selfies=raw_selfies,
                resulting_smiles=resulting_smiles,
                inchi_key_connectivity=result_key,
                accepted=was_accepted,
                rejection_reason=rejection_reason,
            )
        )

    stats.accepted_unique = len(accepted)
    return accepted, proposals, stats
