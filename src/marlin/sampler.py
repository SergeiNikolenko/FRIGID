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


@dataclass(frozen=True)
class MarlinGenerationStats:
    attempts: int
    valid: int
    mass_valid: int
    unique_mass_valid: int


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
        forbidden_token_ids: Sequence[int] = (),
    ) -> None:
        self.model = model
        self.constraint = constraint
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id
        self.decode_tokens = decode_tokens
        self.safe_to_smiles = safe_to_smiles
        self.grammar_mask = grammar_mask
        self.forbidden_token_ids = tuple(forbidden_token_ids)

    def _decode_prefix(self, token_ids: Sequence[int]) -> str:
        try:
            end = token_ids.index(self.eos_token_id) + 1
        except ValueError:
            end = len(token_ids)
        return self.decode_tokens(token_ids[:end])

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

            safe = self._decode_prefix(prefix)
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
        ranked, _ = self.generate_ranked_with_stats(
            fingerprint,
            target_mass,
            candidates=candidates,
            diversity_dropout=diversity_dropout,
            temperature=temperature,
            generator=generator,
        )
        return ranked

    @torch.no_grad()
    def generate_ranked_with_stats(
        self,
        fingerprint: torch.Tensor,
        target_mass: float,
        *,
        candidates: int = 384,
        diversity_dropout: float = 0.3,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> tuple[list[MarlinCandidate], MarlinGenerationStats]:
        if candidates <= 0:
            raise ValueError("candidates must be positive")
        original = (fingerprint > 0.5).to(torch.float32)
        generated, valid = self._generate_many(
            original,
            target_mass,
            candidates=candidates,
            diversity_dropout=diversity_dropout,
            temperature=temperature,
            generator=generator,
        )
        unique: dict[str, tuple[str, str]] = {}
        mass_valid = 0
        for result in generated:
            if result is None:
                continue
            mass_valid += 1
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
        ranked = sorted(ranked, key=lambda candidate: (-candidate.tanimoto, abs(candidate.mass_error_ppm)))
        return ranked, MarlinGenerationStats(
            attempts=candidates,
            valid=valid,
            mass_valid=mass_valid,
            unique_mass_valid=len(ranked),
        )

    def _generate_many(
        self,
        fingerprint: torch.Tensor,
        target_mass: float,
        *,
        candidates: int,
        diversity_dropout: float,
        temperature: float,
        generator: torch.Generator | None,
    ) -> tuple[list[tuple[str, str] | None], int]:
        """Generate candidates in one GPU batch and retain per-row constraints."""
        device = next(self.model.parameters()).device
        conditioned = torch.stack(
            [
                perturb_fingerprint(
                    fingerprint, dropout=diversity_dropout, generator=generator
                )
                for _ in range(candidates)
            ]
        ).to(device=device, dtype=torch.float32)
        masses = torch.full((candidates,), target_mass, device=device)
        prefix = torch.full(
            (candidates, 1), self.bos_token_id, device=device, dtype=torch.long
        )
        states = [MassShellState() for _ in range(candidates)]
        active = torch.ones(candidates, dtype=torch.bool, device=device)
        results: list[tuple[str, str] | None] = [None] * candidates
        valid = 0
        max_blocks = (self.model.config.max_length - 1) // self.model.config.block_width

        for _ in range(max_blocks):
            if not active.any():
                break
            block_start = prefix.shape[1]
            masks = torch.full(
                (candidates, self.model.config.block_width),
                self.mask_token_id,
                device=device,
                dtype=torch.long,
            )
            prefix = torch.cat((prefix, masks), dim=1)
            unresolved = torch.ones(
                (candidates, self.model.config.block_width),
                dtype=torch.bool,
                device=device,
            )
            unresolved[~active] = False

            for _ in range(self.model.config.block_width):
                if not unresolved.any():
                    break
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    logits = self.model(prefix, masses, conditioned)
                for row in torch.nonzero(active, as_tuple=False).flatten().tolist():
                    positions = torch.nonzero(unresolved[row], as_tuple=False).flatten()
                    if positions.numel() == 0:
                        continue
                    best_position = None
                    best_token = None
                    best_confidence = -torch.inf
                    for relative_position in positions.tolist():
                        position = block_start + relative_position
                        position_logits = self.constraint.apply(
                            logits[row, position] / temperature,
                            states[row],
                            target_mass,
                        )
                        if self.forbidden_token_ids:
                            position_logits[list(self.forbidden_token_ids)] = -torch.inf
                        if self.grammar_mask is not None:
                            position_logits = self.grammar_mask(
                                prefix[row, :position].tolist(), position_logits
                            )
                        probabilities = position_logits.softmax(dim=-1)
                        confidence, token = probabilities.max(dim=-1)
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_position = relative_position
                            best_token = int(token)
                    if best_position is None or not torch.isfinite(best_confidence):
                        active[row] = False
                        unresolved[row] = False
                        continue
                    prefix[row, block_start + best_position] = best_token
                    states[row] = self.constraint.advance(states[row], best_token)
                    unresolved[row, best_position] = False

            for row in torch.nonzero(active, as_tuple=False).flatten().tolist():
                safe = self._decode_prefix(prefix[row].tolist())
                smiles = self.safe_to_smiles(safe)
                molecule = Chem.MolFromSmiles(smiles) if smiles else None
                if self.constraint.accepts_smiles(smiles, target_mass):
                    valid += 1
                    results[row] = (safe, smiles)
                    active[row] = False
                elif self.eos_token_id in prefix[row, block_start:].tolist():
                    valid += int(molecule is not None)
                    active[row] = False

        for row in torch.nonzero(active, as_tuple=False).flatten().tolist():
            safe = self._decode_prefix(prefix[row].tolist())
            smiles = self.safe_to_smiles(safe)
            valid += int(bool(smiles) and Chem.MolFromSmiles(smiles) is not None)
        return results, valid


def _morgan(molecule: Chem.Mol):
    return AllChem.GetMorganGenerator(radius=2, fpSize=4096).GetFingerprint(molecule)


def _fingerprint(array: torch.Tensor):
    bit_vector = DataStructs.ExplicitBitVect(array.numel())
    for index in torch.nonzero(array.reshape(-1), as_tuple=False).flatten().tolist():
        bit_vector.SetBit(index)
    return bit_vector
