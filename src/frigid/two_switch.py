"""Deterministic, chemistry-validated 2-switch molecular graph neighbors."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
import random
from typing import Iterable

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


Bond = tuple[int, int]


@dataclass(frozen=True)
class TwoSwitchProposal:
    """One degree-preserving rewiring of two disjoint equal-order bonds."""

    removed_bonds: tuple[Bond, Bond]
    added_bonds: tuple[Bond, Bond]
    bond_type: Chem.BondType
    aromatic: bool


@dataclass(frozen=True)
class TwoSwitchNeighbor:
    """A validated unique molecular neighbor produced by a 2-switch."""

    smiles: str
    inchi_key_connectivity: str
    removed_bonds: tuple[Bond, Bond]
    added_bonds: tuple[Bond, Bond]


@dataclass
class TwoSwitchStatistics:
    """Proposal-level accounting for a bounded neighbor-generation call."""

    proposals_available: int = 0
    proposals_considered: int = 0
    accepted_unique: int = 0
    invalid_counts: Counter[str] = field(default_factory=Counter)

    def as_dict(self) -> dict[str, int | dict[str, int]]:
        return {
            "proposals_available": self.proposals_available,
            "proposals_considered": self.proposals_considered,
            "accepted_unique": self.accepted_unique,
            "invalid_counts": dict(sorted(self.invalid_counts.items())),
        }


def _ordered_bond(atom_a: int, atom_b: int) -> Bond:
    return (atom_a, atom_b) if atom_a < atom_b else (atom_b, atom_a)


def _atom_signature(mol: Chem.Mol) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (atom.GetAtomicNum(), atom.GetFormalCharge(), atom.GetIsotope())
        for atom in mol.GetAtoms()
    )


def _degree_vector(mol: Chem.Mol) -> tuple[int, ...]:
    return tuple(atom.GetDegree() for atom in mol.GetAtoms())


def _valence_vector(mol: Chem.Mol) -> tuple[int, ...]:
    return tuple(atom.GetTotalValence() for atom in mol.GetAtoms())


def _connectivity_key(mol: Chem.Mol) -> str | None:
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    if not key:
        return None
    return key.split("-", maxsplit=1)[0]


def _prepare_seed(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid seed SMILES: {smiles}")
    Chem.SanitizeMol(mol)
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError(f"Seed must be connected: {smiles}")
    return mol


def enumerate_two_switch_proposals(mol: Chem.Mol) -> list[TwoSwitchProposal]:
    """Enumerate all endpoint pairings for eligible bond pairs."""

    bonds = []
    for bond in mol.GetBonds():
        edge = _ordered_bond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        bonds.append((edge, bond.GetBondType(), bond.GetIsAromatic()))
    bonds.sort(key=lambda item: (item[0], str(item[1]), item[2]))

    proposals: list[TwoSwitchProposal] = []
    for (edge_a, type_a, aromatic_a), (edge_b, type_b, aromatic_b) in combinations(
        bonds, 2
    ):
        if type_a != type_b or aromatic_a != aromatic_b:
            continue
        if set(edge_a).intersection(edge_b):
            continue

        atom_a, atom_b = edge_a
        atom_c, atom_d = edge_b
        for added in (
            ((_ordered_bond(atom_a, atom_c)), (_ordered_bond(atom_b, atom_d))),
            ((_ordered_bond(atom_a, atom_d)), (_ordered_bond(atom_b, atom_c))),
        ):
            proposals.append(
                TwoSwitchProposal(
                    removed_bonds=(edge_a, edge_b),
                    added_bonds=added,
                    bond_type=type_a,
                    aromatic=aromatic_a,
                )
            )
    return proposals


def _apply_proposal(mol: Chem.Mol, proposal: TwoSwitchProposal) -> Chem.Mol:
    rw_mol = Chem.RWMol(mol)
    for atom_a, atom_b in proposal.removed_bonds:
        rw_mol.RemoveBond(atom_a, atom_b)
    for atom_a, atom_b in proposal.added_bonds:
        rw_mol.AddBond(atom_a, atom_b, proposal.bond_type)
        if proposal.aromatic:
            new_bond = rw_mol.GetBondBetweenAtoms(atom_a, atom_b)
            assert new_bond is not None
            new_bond.SetIsAromatic(True)
    return rw_mol.GetMol()


def generate_two_switch_neighbors(
    smiles: str,
    *,
    seed: int,
    max_proposals: int,
    max_neighbors: int,
    exclude_connectivity_keys: Iterable[str] = (),
) -> tuple[list[TwoSwitchNeighbor], TwoSwitchStatistics]:
    """Generate a bounded deterministic set of validated one-step neighbors."""

    if max_proposals <= 0:
        raise ValueError("max_proposals must be positive")
    if max_neighbors <= 0:
        raise ValueError("max_neighbors must be positive")

    mol = _prepare_seed(smiles)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    atom_signature = _atom_signature(mol)
    degree_vector = _degree_vector(mol)
    valence_vector = _valence_vector(mol)
    original_edges = {
        _ordered_bond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in mol.GetBonds()
    }
    seed_key = _connectivity_key(mol)
    excluded = set(exclude_connectivity_keys)
    if seed_key is not None:
        excluded.add(seed_key)

    proposals = enumerate_two_switch_proposals(mol)
    random.Random(seed).shuffle(proposals)
    stats = TwoSwitchStatistics(proposals_available=len(proposals))
    neighbors: list[TwoSwitchNeighbor] = []

    for proposal in proposals[:max_proposals]:
        if len(neighbors) >= max_neighbors:
            break
        stats.proposals_considered += 1

        remaining_edges = original_edges.difference(proposal.removed_bonds)
        added_a, added_b = proposal.added_bonds
        if (
            added_a == added_b
            or added_a in remaining_edges
            or added_b in remaining_edges
            or added_a[0] == added_a[1]
            or added_b[0] == added_b[1]
        ):
            stats.invalid_counts["existing_or_duplicate_edge"] += 1
            continue

        try:
            candidate = _apply_proposal(mol, proposal)
            Chem.SanitizeMol(candidate)
        except Exception:
            stats.invalid_counts["sanitize_failure"] += 1
            continue

        if len(Chem.GetMolFrags(candidate)) != 1:
            stats.invalid_counts["disconnected"] += 1
            continue
        if (
            candidate.GetNumAtoms() != mol.GetNumAtoms()
            or _atom_signature(candidate) != atom_signature
        ):
            stats.invalid_counts["atom_change"] += 1
            continue
        if rdMolDescriptors.CalcMolFormula(candidate) != formula:
            stats.invalid_counts["formula_change"] += 1
            continue
        if _degree_vector(candidate) != degree_vector:
            stats.invalid_counts["degree_change"] += 1
            continue
        if _valence_vector(candidate) != valence_vector:
            stats.invalid_counts["valence_change"] += 1
            continue

        Chem.RemoveStereochemistry(candidate)
        candidate_key = _connectivity_key(candidate)
        if candidate_key is None:
            stats.invalid_counts["inchikey_failure"] += 1
            continue
        if candidate_key in excluded:
            stats.invalid_counts["duplicate_connectivity"] += 1
            continue

        candidate_smiles = Chem.MolToSmiles(
            candidate, canonical=True, isomericSmiles=False
        )
        excluded.add(candidate_key)
        neighbors.append(
            TwoSwitchNeighbor(
                smiles=candidate_smiles,
                inchi_key_connectivity=candidate_key,
                removed_bonds=proposal.removed_bonds,
                added_bonds=proposal.added_bonds,
            )
        )

    stats.accepted_unique = len(neighbors)
    return neighbors, stats
