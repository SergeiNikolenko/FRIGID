"""Natural M+1 and M+2 isotope-envelope ratios for training conditioning."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache

import torch
from rdkit import Chem


@lru_cache(maxsize=4096)
def _ratios_from_atom_counts(atom_counts: tuple[tuple[int, int], ...]) -> tuple[float, float]:
    periodic_table = Chem.GetPeriodicTable()
    coefficient_1 = 0.0
    coefficient_2 = 0.0
    for atomic_number, count in atom_counts:
        base_isotope = periodic_table.GetMostCommonIsotope(atomic_number)
        base_abundance = periodic_table.GetAbundanceForIsotope(
            atomic_number, base_isotope
        )
        if base_abundance <= 0:
            continue
        ratio_1 = (
            periodic_table.GetAbundanceForIsotope(atomic_number, base_isotope + 1)
            / base_abundance
        )
        ratio_2 = (
            periodic_table.GetAbundanceForIsotope(atomic_number, base_isotope + 2)
            / base_abundance
        )
        for _ in range(count):
            coefficient_2 = (
                coefficient_2 + coefficient_1 * ratio_1 + ratio_2
            )
            coefficient_1 = coefficient_1 + ratio_1
    return coefficient_1, coefficient_2


def theoretical_isotope_ratios(molecule: Chem.Mol) -> torch.Tensor:
    """Return theoretical natural-abundance ``[M+1/M, M+2/M]`` ratios.

    The paper does not disclose how its training isotope ratios were obtained.
    This clean-room implementation uses a nominal-mass, degree-two convolution
    of RDKit natural isotope abundances and is therefore an inferred choice.
    """
    atom_counts: Counter[int] = Counter()
    hydrogen_count = 0
    for atom in molecule.GetAtoms():
        atom_counts[atom.GetAtomicNum()] += 1
        if atom.GetAtomicNum() != 1:
            hydrogen_count += atom.GetTotalNumHs(includeNeighbors=False)
    atom_counts[1] += hydrogen_count
    ratios = _ratios_from_atom_counts(tuple(sorted(atom_counts.items())))
    return torch.tensor(ratios, dtype=torch.float32)
