"""Formula-free mass-shell constraints for SAFE token decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors


HYDROGEN_MASS = 1.00782503223


@dataclass(frozen=True)
class MassShellState:
    heavy_mass: float = 0.0
    heavy_atoms: int = 0
    valence_sum: float = 0.0


class MassShellConstraint:
    def __init__(
        self,
        token_masses: Sequence[float],
        token_heavy_atoms: Sequence[int] | None = None,
        token_valences: Sequence[float] | None = None,
        *,
        ppm_tolerance: float = 10.0,
        valence_slack: float = 4.0,
        eos_boost: float = 1.0,
        eos_token_id: int,
    ) -> None:
        if ppm_tolerance <= 0:
            raise ValueError("ppm_tolerance must be positive")
        self.token_masses = torch.as_tensor(token_masses, dtype=torch.float64)
        size = self.token_masses.numel()
        self.token_heavy_atoms = torch.as_tensor(
            token_heavy_atoms if token_heavy_atoms is not None else [0] * size,
            dtype=torch.int64,
        )
        self.token_valences = torch.as_tensor(
            token_valences if token_valences is not None else [0.0] * size,
            dtype=torch.float64,
        )
        if self.token_heavy_atoms.numel() != size or self.token_valences.numel() != size:
            raise ValueError("token property arrays must have equal lengths")
        self.ppm_tolerance = ppm_tolerance
        self.valence_slack = valence_slack
        self.eos_boost = eos_boost
        self.eos_token_id = eos_token_id
        positive_masses = self.token_masses[self.token_masses > 0]
        self.minimum_token_mass = (
            float(positive_masses.min()) if positive_masses.numel() else float("inf")
        )

    def tolerance(self, target_mass: float) -> float:
        return self.ppm_tolerance * 1e-6 * target_mass

    def advance(self, state: MassShellState, token_id: int) -> MassShellState:
        return MassShellState(
            heavy_mass=state.heavy_mass + float(self.token_masses[token_id]),
            heavy_atoms=state.heavy_atoms + int(self.token_heavy_atoms[token_id]),
            valence_sum=state.valence_sum + float(self.token_valences[token_id]),
        )

    def hydrogen_capacity(self, state: MassShellState) -> float:
        skeleton_bonds = max(2 * (state.heavy_atoms - 1), 0)
        return max(state.valence_sum - skeleton_bonds + self.valence_slack, 0.0)

    def apply(self, logits: torch.Tensor, state: MassShellState, target_mass: float) -> torch.Tensor:
        """Apply the safe upper prune and conservative EOS coupling."""
        if logits.shape[-1] != self.token_masses.numel():
            raise ValueError("logit vocabulary does not match token mass table")
        constrained = logits.clone()
        masses = self.token_masses.to(logits.device)
        delta = self.tolerance(target_mass)
        constrained[..., state.heavy_mass + masses > target_mass + delta] = -torch.inf
        reachable = state.heavy_mass + self.hydrogen_capacity(state) * HYDROGEN_MASS
        if reachable < target_mass - delta:
            constrained[..., self.eos_token_id] = -torch.inf
        elif state.heavy_mass + self.minimum_token_mass > target_mass + delta:
            constrained[..., self.eos_token_id] += self.eos_boost
        return constrained

    def accepts_smiles(self, smiles: str | None, target_mass: float) -> bool:
        if not smiles:
            return False
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return False
        exact_mass = Descriptors.ExactMolWt(molecule)
        return abs(exact_mass - target_mass) <= self.tolerance(target_mass)
