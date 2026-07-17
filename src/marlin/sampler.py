"""Mass-shell constrained block-diffusion sampling and ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

from marlin.mass_shell import MassShellConstraint, MassShellState
from marlin.model import MarlinDecoder
from marlin.noise import perturb_fingerprint


@dataclass(frozen=True)
class MarlinCandidate:
    smiles: str
    safe: str
    tanimoto: float
    mass_error_ppm: float


class MarlinSampler:
    def __init__(
        self,
        model: MarlinDecoder,
        constraint: MassShellConstraint,
        *,
        bos_token_id: int,
        eos_token_id: int,
        mask_token_id: int,
        decode_tokens: Callable[[Sequence[int]], str],
        safe_to_smiles: Callable[[str], str | None],
        grammar_mask: Callable[[Sequence[int], torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.model = model
        self.constraint = constraint
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id
        self.decode_tokens = decode_tokens
        self.safe_to_smiles = safe_to_smiles
        self.grammar_mask = grammar_mask

    @torch.no_grad()
    def generate_one(
        self,
        fingerprint: torch.Tensor,
        target_mass: float,
        *,
        max_blocks: int | None = None,
        temperature: float = 1.0,
    ) -> tuple[str, str] | None:
        device = next(self.model.parameters()).device
        fingerprint = fingerprint.to(device=device, dtype=torch.float32).reshape(1, -1)
        mass = torch.tensor([target_mass], device=device)
        prefix = [self.bos_token_id]
        state = MassShellState()
        max_blocks = max_blocks or (self.model.config.max_length - 1) // self.model.config.block_width

        for _ in range(max_blocks):
            block_start = len(prefix)
            prefix.extend([self.mask_token_id] * self.model.config.block_width)
            if len(prefix) > self.model.config.max_length:
                return None
            unresolved = set(range(block_start, len(prefix)))
            while unresolved:
                input_ids = torch.tensor([prefix], device=device)
                logits = self.model(input_ids, mass, fingerprint)[0]
                best_position = None
                best_token = None
                best_confidence = -torch.inf
                for position in unresolved:
                    position_logits = self.constraint.apply(logits[position] / temperature, state, target_mass)
                    if self.grammar_mask is not None:
                        position_logits = self.grammar_mask(prefix[:position], position_logits)
                    probabilities = position_logits.softmax(dim=-1)
                    confidence, token = probabilities.max(dim=-1)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_position = position
                        best_token = int(token)
                if best_position is None or not torch.isfinite(best_confidence):
                    return None
                prefix[best_position] = best_token
                state = self.constraint.advance(state, best_token)
                unresolved.remove(best_position)

            safe = self.decode_tokens(prefix)
            smiles = self.safe_to_smiles(safe)
            if self.constraint.accepts_smiles(smiles, target_mass):
                return safe, smiles
            if self.eos_token_id in prefix[block_start:]:
                return None
        return None

    @torch.no_grad()
    def generate_ranked(
        self,
        fingerprint: torch.Tensor,
        target_mass: float,
        *,
        candidates: int = 384,
        diversity_dropout: float = 0.3,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> list[MarlinCandidate]:
        original = (fingerprint > 0.5).to(torch.float32)
        unique: dict[str, tuple[str, str]] = {}
        for _ in range(candidates):
            conditioned = perturb_fingerprint(original, dropout=diversity_dropout, generator=generator)
            result = self.generate_one(conditioned, target_mass, temperature=temperature)
            if result is not None:
                safe, smiles = result
                molecule = Chem.MolFromSmiles(smiles)
                if molecule is not None:
                    canonical = Chem.MolToSmiles(molecule, canonical=True)
                    unique.setdefault(canonical, (safe, canonical))

        reference = _fingerprint(original)
        ranked = []
        for safe, smiles in unique.values():
            molecule = Chem.MolFromSmiles(smiles)
            exact_mass = Descriptors.ExactMolWt(molecule)
            ranked.append(
                MarlinCandidate(
                    smiles=smiles,
                    safe=safe,
                    tanimoto=DataStructs.TanimotoSimilarity(reference, _morgan(molecule)),
                    mass_error_ppm=1e6 * (exact_mass - target_mass) / target_mass,
                )
            )
        return sorted(ranked, key=lambda candidate: (-candidate.tanimoto, abs(candidate.mass_error_ppm)))


def _morgan(molecule: Chem.Mol):
    return AllChem.GetMorganGenerator(radius=2, fpSize=4096).GetFingerprint(molecule)


def _fingerprint(array: torch.Tensor):
    bit_vector = DataStructs.ExplicitBitVect(array.numel())
    for index in torch.nonzero(array.reshape(-1), as_tuple=False).flatten().tolist():
        bit_vector.SetBit(index)
    return bit_vector
