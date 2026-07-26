"""Mass-shell constrained block-diffusion sampling and ranking."""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    strict_valid: int
    mass_valid: int
    unique_mass_valid: int
    constraint_dead_ends: int
    eos_terminated: int
    max_length_terminated: int
    sample_terminal_safes: tuple[str, ...]
    sample_dead_ends: tuple[dict[str, float | int | str], ...]


@dataclass(frozen=True)
class MarlinBeamSearchStats:
    beam_width: int
    branch_factor: int
    max_model_batch_size: int
    max_completed_paths: int
    expanded_hypotheses: int
    expanded_tokens: int
    completed_paths: int
    valid_paths: int
    strict_valid_paths: int
    mass_valid_paths: int
    constraint_dead_ends: int
    eos_terminated: int
    block_terminated: int
    max_length_terminated: int
    pruned_hypotheses: int
    backtrack_recoveries: int
    best_completed_log_probability: float | None
    max_committed_tokens: int
    sample_terminal_safes: tuple[str, ...]
    sample_dead_ends: tuple[dict[str, float | int | str], ...]
    completion_paths: tuple[dict[str, float | int | bool | str | None], ...]


@dataclass(frozen=True)
class _BeamHypothesis:
    canvas: tuple[int, ...]
    state: MassShellState
    log_probability: float
    used_alternative: bool


@dataclass(frozen=True)
class _BeamCompletion:
    canvas: tuple[int, ...]
    log_probability: float
    termination: str
    used_alternative: bool


@dataclass(frozen=True)
class _BeamExpansion:
    log_probability: float
    hypothesis: _BeamHypothesis | None = None
    completion: _BeamCompletion | None = None


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
        strict_safe_to_smiles: Callable[[str], str | None] | None = None,
        grammar_mask: Callable[
            [Sequence[int], torch.Tensor, float | None], torch.Tensor
        ]
        | None = None,
        forbidden_token_ids: Sequence[int] = (),
        mass_shell_enabled: bool = True,
        generation_mode: str = "block",
    ) -> None:
        if generation_mode not in {"block", "canvas"}:
            raise ValueError("generation_mode must be 'block' or 'canvas'")
        self.model = model
        self.constraint = constraint
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id
        self.decode_tokens = decode_tokens
        self.safe_to_smiles = safe_to_smiles
        self.strict_safe_to_smiles = strict_safe_to_smiles or safe_to_smiles
        self.grammar_mask = grammar_mask
        self.forbidden_token_ids = tuple(forbidden_token_ids)
        self.mass_shell_enabled = mass_shell_enabled
        self.generation_mode = generation_mode

    def _sampling_logits(
        self,
        input_ids: torch.Tensor,
        precursor_mass: torch.Tensor,
        fingerprint: torch.Tensor,
    ) -> torch.Tensor:
        sampling_logits = getattr(self.model, "sampling_logits", None)
        if sampling_logits is not None:
            return sampling_logits(input_ids, precursor_mass, fingerprint)
        return self.model(input_ids, precursor_mass, fingerprint)

    def _decode_prefix(self, token_ids: Sequence[int]) -> str:
        try:
            end = token_ids.index(self.eos_token_id) + 1
        except ValueError:
            end = len(token_ids)
        return self.decode_tokens(token_ids[:end])

    def _mass_state(self, token_ids: Sequence[int]) -> MassShellState:
        """Recompute mass from every committed token before EOS."""
        state = MassShellState()
        for token_id in token_ids[1:]:
            if token_id == self.eos_token_id:
                break
            if token_id == self.mask_token_id:
                continue
            state = self.constraint.advance(state, token_id)
        return state

    def constrain_action_logits(
        self,
        prefix_ids: Sequence[int],
        logits: torch.Tensor,
        target_mass: float,
        *,
        state: MassShellState | None = None,
    ) -> torch.Tensor:
        """Apply the production constraints for one proposed token action."""

        constrained = logits.clone()
        if self.mass_shell_enabled:
            constrained = self.constraint.apply(
                constrained,
                state if state is not None else self._mass_state(prefix_ids),
                target_mass,
                allow_early_eos=self.mask_token_id in prefix_ids,
            )
        if self.forbidden_token_ids:
            constrained[list(self.forbidden_token_ids)] = -torch.inf
        if self.grammar_mask is not None:
            constrained = self.grammar_mask(
                prefix_ids,
                constrained,
                target_mass,
            )
        return constrained

    def _next_block_width(self, prefix_length: int) -> int:
        remaining = self.model.config.max_length - prefix_length
        content_length = prefix_length - 1
        offset = content_length % self.model.config.block_width
        aligned_width = (
            self.model.config.block_width - offset
            if offset
            else self.model.config.block_width
        )
        return min(aligned_width, remaining)

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
        blocks_remaining = max_blocks
        while len(prefix) < self.model.config.max_length:
            if blocks_remaining is not None:
                if blocks_remaining <= 0:
                    break
                blocks_remaining -= 1
            block_start = len(prefix)
            block_width = self._next_block_width(block_start)
            prefix.extend([self.mask_token_id] * block_width)
            unresolved = set(range(block_start, len(prefix)))
            while unresolved:
                input_ids = torch.tensor([prefix], device=device)
                logits = self._sampling_logits(input_ids, mass, fingerprint)[0]
                best_position = None
                best_token = None
                best_confidence = -torch.inf
                positions = sorted(unresolved)
                for position in positions:
                    position_logits = self.constrain_action_logits(
                        prefix[:position],
                        logits[position] / temperature,
                        target_mass,
                        state=state,
                    )
                    probabilities = position_logits.softmax(dim=-1)
                    confidence, token = probabilities.max(dim=-1)
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_position = position
                        best_token = int(token)
                if best_position is None or not torch.isfinite(best_confidence):
                    return None
                prefix[best_position] = best_token
                unresolved.remove(best_position)
                if best_token == self.eos_token_id:
                    for position in tuple(unresolved):
                        if position > best_position:
                            prefix[position] = self.mask_token_id
                            unresolved.remove(position)
                state = self._mass_state(prefix)

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
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        original = (fingerprint > 0.5).to(torch.float32)
        generator_fn = (
            self._generate_many_canvas
            if self.generation_mode == "canvas"
            else self._generate_many
        )
        generated, valid, diagnostics = generator_fn(
            original,
            target_mass,
            candidates=candidates,
            diversity_dropout=diversity_dropout,
            temperature=temperature,
            generator=generator,
        )
        unique: dict[str, tuple[str, str, bool]] = {}
        mass_valid = 0
        for result in generated:
            if result is None:
                continue
            safe, smiles, is_mass_valid = result
            mass_valid += int(is_mass_valid)
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is not None:
                canonical = Chem.MolToSmiles(molecule, canonical=True)
                unique.setdefault(canonical, (safe, canonical, is_mass_valid))

        reference = _fingerprint(original)
        ranked = []
        unique_mass_valid = 0
        for safe, smiles, is_mass_valid in unique.values():
            unique_mass_valid += int(is_mass_valid)
            molecule = Chem.MolFromSmiles(smiles)
            exact_mass = Descriptors.ExactMolWt(molecule)
            ranked.append(
                MarlinCandidate(
                    smiles=smiles,
                    safe=safe,
                    tanimoto=DataStructs.TanimotoSimilarity(
                        reference, _morgan(molecule)
                    ),
                    mass_error_ppm=1e6 * (exact_mass - target_mass) / target_mass,
                )
            )
        ranked = sorted(
            ranked,
            key=lambda candidate: (-candidate.tanimoto, abs(candidate.mass_error_ppm)),
        )
        return ranked, MarlinGenerationStats(
            attempts=candidates,
            valid=valid,
            strict_valid=int(diagnostics["strict_valid"]),
            mass_valid=mass_valid,
            unique_mass_valid=unique_mass_valid,
            constraint_dead_ends=diagnostics["constraint_dead_ends"],
            eos_terminated=diagnostics["eos_terminated"],
            max_length_terminated=diagnostics["max_length_terminated"],
            sample_terminal_safes=tuple(diagnostics["sample_terminal_safes"]),
            sample_dead_ends=tuple(diagnostics["sample_dead_ends"]),
        )

    @torch.no_grad()
    def generate_beam_ranked_with_stats(
        self,
        fingerprint: torch.Tensor,
        target_mass: float,
        *,
        beam_width: int = 64,
        branch_factor: int = 10,
        temperature: float = 1.0,
        max_model_batch_size: int = 64,
        max_completed_paths: int | None = None,
    ) -> tuple[list[MarlinCandidate], MarlinBeamSearchStats]:
        """Search contiguous constrained prefixes without irreversible argmax.

        This is a bounded diagnostic recovery path for models trained on the
        first unresolved token. It leaves the production greedy sampler
        unchanged and keeps the best active prefixes by cumulative log
        probability after every committed token.
        """

        if beam_width <= 0:
            raise ValueError("beam_width must be positive")
        if branch_factor <= 0:
            raise ValueError("branch_factor must be positive")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(target_mass) or target_mass <= 0:
            raise ValueError("target_mass must be finite and positive")
        if max_model_batch_size <= 0:
            raise ValueError("max_model_batch_size must be positive")
        completion_limit = (
            beam_width if max_completed_paths is None else max_completed_paths
        )
        if completion_limit <= 0:
            raise ValueError("max_completed_paths must be positive")
        if self.generation_mode != "block":
            raise ValueError("beam search currently requires block generation mode")

        device = next(self.model.parameters()).device
        original = (fingerprint > 0.5).to(device=device, dtype=torch.float32)
        conditioned = original.reshape(1, -1)
        first_width = self._next_block_width(1)
        if first_width <= 0:
            raise ValueError(
                "beam search requires model.config.max_length to be at least 2"
            )
        beams = [
            _BeamHypothesis(
                canvas=(self.bos_token_id,) + (self.mask_token_id,) * first_width,
                state=MassShellState(),
                log_probability=0.0,
                used_alternative=False,
            )
        ]
        completions: list[_BeamCompletion] = []
        expanded_hypotheses = 0
        expanded_tokens = 0
        constraint_dead_ends = 0
        eos_terminated = 0
        block_terminated = 0
        max_length_terminated = 0
        pruned_hypotheses = 0
        max_committed_tokens = 0
        dead_end_examples: list[dict[str, float | int | str]] = []

        def content_token_count(canvas: Sequence[int]) -> int:
            count = 0
            for token_id in canvas[1:]:
                if token_id in {self.eos_token_id, self.mask_token_id}:
                    break
                count += 1
            return count

        def production_accepts(canvas: Sequence[int]) -> bool:
            safe = self._decode_prefix(canvas)
            smiles = self.safe_to_smiles(safe)
            molecule = Chem.MolFromSmiles(smiles) if smiles else None
            is_valid = molecule is not None
            is_mass_valid = self.constraint.accepts_smiles(smiles, target_mass)
            return is_mass_valid or (is_valid and not self.mass_shell_enabled)

        while beams:
            canvas_lengths = {len(beam.canvas) for beam in beams}
            if len(canvas_lengths) != 1:
                raise AssertionError("active beam canvases must have equal lengths")
            try:
                position = beams[0].canvas.index(self.mask_token_id)
            except ValueError as error:
                raise AssertionError("active beam has no unresolved token") from error
            if any(beam.canvas.index(self.mask_token_id) != position for beam in beams):
                raise AssertionError(
                    "active beam canvases must share the unresolved position"
                )

            token_logit_chunks = []
            for start in range(0, len(beams), max_model_batch_size):
                chunk = beams[start : start + max_model_batch_size]
                input_ids = torch.tensor(
                    [beam.canvas for beam in chunk],
                    device=device,
                    dtype=torch.long,
                )
                masses = torch.full(
                    (len(chunk),), target_mass, device=device, dtype=torch.float32
                )
                fingerprints = conditioned.expand(len(chunk), -1)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    logits = self._sampling_logits(input_ids, masses, fingerprints)
                token_logit_chunks.append(logits[:, position].float())
            token_logits = torch.cat(token_logit_chunks, dim=0)

            expansions: list[_BeamExpansion] = []
            constrained_rows = []
            for beam, row_logits in zip(beams, token_logits):
                constrained_rows.append(
                    self.constrain_action_logits(
                        beam.canvas[:position],
                        row_logits / temperature,
                        target_mass,
                        state=beam.state,
                    )
                )
            constrained_logits = torch.stack(constrained_rows)
            finite = torch.isfinite(constrained_logits)
            usable = constrained_logits.masked_fill(~finite, -torch.inf)
            log_normalizer = torch.logsumexp(usable, dim=-1, keepdim=True)
            log_probabilities = usable - log_normalizer
            log_probabilities = torch.where(
                finite.any(dim=-1, keepdim=True),
                log_probabilities,
                torch.full_like(log_probabilities, -torch.inf),
            )
            selected_width = min(branch_factor, log_probabilities.shape[-1])
            ordered_ids = torch.argsort(
                log_probabilities,
                dim=-1,
                descending=True,
                stable=True,
            )[:, :selected_width]
            selected_values = torch.gather(log_probabilities, dim=-1, index=ordered_ids)
            packed = (
                torch.stack(
                    (selected_values, ordered_ids.to(selected_values.dtype)),
                    dim=-1,
                )
                .cpu()
                .tolist()
            )

            expanded_hypotheses += len(beams)
            for beam, row_choices in zip(beams, packed):
                finite_choices = [
                    (float(value), int(token_id))
                    for value, token_id in row_choices
                    if math.isfinite(float(value))
                ]
                if not finite_choices:
                    constraint_dead_ends += 1
                    if len(dead_end_examples) < 5:
                        dead_end_examples.append(
                            {
                                "safe": self._decode_prefix(beam.canvas)[:512],
                                "heavy_mass": beam.state.heavy_mass,
                                "heavy_atoms": beam.state.heavy_atoms,
                                "valence_sum": beam.state.valence_sum,
                            }
                        )
                    continue

                for local_rank, (value, token_id) in enumerate(finite_choices):
                    expanded_tokens += 1
                    canvas = list(beam.canvas)
                    canvas[position] = token_id
                    score = beam.log_probability + value
                    used_alternative = beam.used_alternative or local_rank > 0
                    if token_id == self.eos_token_id:
                        eos_terminated += 1
                        if production_accepts(canvas):
                            expansions.append(
                                _BeamExpansion(
                                    log_probability=score,
                                    completion=_BeamCompletion(
                                        canvas=tuple(canvas),
                                        log_probability=score,
                                        termination="eos",
                                        used_alternative=used_alternative,
                                    ),
                                )
                            )
                        continue

                    max_committed_tokens = max(
                        max_committed_tokens,
                        content_token_count(canvas),
                    )
                    state = self.constraint.advance(beam.state, token_id)
                    if self.mask_token_id not in canvas:
                        safe = self._decode_prefix(canvas)
                        smiles = self.safe_to_smiles(safe)
                        is_valid = bool(
                            smiles and Chem.MolFromSmiles(smiles) is not None
                        )
                        is_mass_valid = self.constraint.accepts_smiles(
                            smiles, target_mass
                        )
                        if is_mass_valid or (is_valid and not self.mass_shell_enabled):
                            block_terminated += 1
                            expansions.append(
                                _BeamExpansion(
                                    log_probability=score,
                                    completion=_BeamCompletion(
                                        canvas=tuple(canvas),
                                        log_probability=score,
                                        termination="block",
                                        used_alternative=used_alternative,
                                    ),
                                )
                            )
                            continue
                        if len(canvas) >= self.model.config.max_length:
                            max_length_terminated += 1
                            continue
                        next_width = self._next_block_width(len(canvas))
                        canvas.extend([self.mask_token_id] * next_width)
                    expansions.append(
                        _BeamExpansion(
                            log_probability=score,
                            hypothesis=_BeamHypothesis(
                                canvas=tuple(canvas),
                                state=state,
                                log_probability=score,
                                used_alternative=used_alternative,
                            ),
                        )
                    )

            new_completions = [
                expansion.completion
                for expansion in expansions
                if expansion.completion is not None
            ]
            completions.extend(new_completions)
            completions.sort(
                key=lambda completion: (
                    -completion.log_probability,
                    completion.canvas,
                )
            )
            if len(completions) > completion_limit:
                pruned_hypotheses += len(completions) - completion_limit
                del completions[completion_limit:]

            live = [
                expansion.hypothesis
                for expansion in expansions
                if expansion.hypothesis is not None
            ]
            live.sort(
                key=lambda hypothesis: (
                    -hypothesis.log_probability,
                    hypothesis.canvas,
                )
            )
            if len(live) > beam_width:
                pruned_hypotheses += len(live) - beam_width
            beams = live[:beam_width]
            if (
                len(completions) == completion_limit
                and beams
                and completions[-1].log_probability >= beams[0].log_probability
            ):
                break

        generated: list[tuple[str, str, bool]] = []
        valid_paths = 0
        strict_valid_paths = 0
        mass_valid_paths = 0
        terminal_examples: list[str] = []
        completion_paths: list[dict[str, float | int | bool | str | None]] = []
        for completion in completions:
            safe = self._decode_prefix(completion.canvas)
            smiles = self.safe_to_smiles(safe)
            molecule = Chem.MolFromSmiles(smiles) if smiles else None
            is_valid = molecule is not None
            strict_smiles = self.strict_safe_to_smiles(safe)
            strict_molecule = (
                Chem.MolFromSmiles(strict_smiles) if strict_smiles else None
            )
            is_mass_valid = self.constraint.accepts_smiles(smiles, target_mass)
            valid_paths += int(is_valid)
            strict_valid_paths += int(strict_molecule is not None)
            mass_valid_paths += int(is_mass_valid)
            if len(terminal_examples) < 5:
                terminal_examples.append(safe[:512])
            completion_paths.append(
                {
                    "log_probability": completion.log_probability,
                    "termination": completion.termination,
                    "used_alternative": completion.used_alternative,
                    "content_tokens": content_token_count(completion.canvas),
                    "safe": safe,
                    "smiles": smiles,
                    "valid": is_valid,
                    "strict_valid": strict_molecule is not None,
                    "mass_valid": is_mass_valid,
                }
            )
            if is_mass_valid or (is_valid and not self.mass_shell_enabled):
                if smiles is None:
                    raise AssertionError("accepted completion has no SMILES")
                generated.append((safe, smiles, is_mass_valid))

        unique: dict[str, tuple[str, str, bool]] = {}
        for safe, smiles, is_mass_valid in generated:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                continue
            canonical = Chem.MolToSmiles(molecule, canonical=True)
            unique.setdefault(canonical, (safe, canonical, is_mass_valid))

        reference = _fingerprint(original)
        ranked = []
        for safe, smiles, _ in unique.values():
            molecule = Chem.MolFromSmiles(smiles)
            exact_mass = Descriptors.ExactMolWt(molecule)
            ranked.append(
                MarlinCandidate(
                    smiles=smiles,
                    safe=safe,
                    tanimoto=DataStructs.TanimotoSimilarity(
                        reference, _morgan(molecule)
                    ),
                    mass_error_ppm=(1e6 * (exact_mass - target_mass) / target_mass),
                )
            )
        ranked.sort(
            key=lambda candidate: (-candidate.tanimoto, abs(candidate.mass_error_ppm))
        )
        return ranked, MarlinBeamSearchStats(
            beam_width=beam_width,
            branch_factor=branch_factor,
            max_model_batch_size=max_model_batch_size,
            max_completed_paths=completion_limit,
            expanded_hypotheses=expanded_hypotheses,
            expanded_tokens=expanded_tokens,
            completed_paths=len(completions),
            valid_paths=valid_paths,
            strict_valid_paths=strict_valid_paths,
            mass_valid_paths=mass_valid_paths,
            constraint_dead_ends=constraint_dead_ends,
            eos_terminated=eos_terminated,
            block_terminated=block_terminated,
            max_length_terminated=max_length_terminated,
            pruned_hypotheses=pruned_hypotheses,
            backtrack_recoveries=sum(
                int(completion.used_alternative) for completion in completions
            ),
            best_completed_log_probability=(
                completions[0].log_probability if completions else None
            ),
            max_committed_tokens=max_committed_tokens,
            sample_terminal_safes=tuple(terminal_examples),
            sample_dead_ends=tuple(dead_end_examples),
            completion_paths=tuple(completion_paths),
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
    ) -> tuple[list[tuple[str, str, bool] | None], int, dict[str, int | list[str]]]:
        """Generate candidates in one GPU batch and retain per-row constraints."""
        device = next(self.model.parameters()).device
        fingerprint = fingerprint.to(device=device, dtype=torch.float32)
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
        results: list[tuple[str, str, bool] | None] = [None] * candidates
        valid = 0
        diagnostics: dict[str, object] = {
            "constraint_dead_ends": 0,
            "eos_terminated": 0,
            "max_length_terminated": 0,
            "strict_valid": 0,
            "sample_terminal_safes": [],
            "sample_dead_ends": [],
        }

        def record_terminal_safe(safe: str) -> None:
            examples = diagnostics["sample_terminal_safes"]
            assert isinstance(examples, list)
            if len(examples) < 5:
                examples.append(safe[:512])

        def record_strict_validity(safe: str) -> None:
            strict_smiles = self.strict_safe_to_smiles(safe)
            if strict_smiles and Chem.MolFromSmiles(strict_smiles) is not None:
                diagnostics["strict_valid"] += 1

        while prefix.shape[1] < self.model.config.max_length:
            if not active.any():
                break
            block_start = prefix.shape[1]
            block_width = self._next_block_width(block_start)
            masks = torch.full(
                (candidates, block_width),
                self.mask_token_id,
                device=device,
                dtype=torch.long,
            )
            prefix = torch.cat((prefix, masks), dim=1)
            unresolved = torch.ones(
                (candidates, block_width),
                dtype=torch.bool,
                device=device,
            )
            unresolved[~active] = False

            for _ in range(block_width):
                if not unresolved.any():
                    break
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    logits = self._sampling_logits(prefix, masses, conditioned)
                for row in torch.nonzero(active, as_tuple=False).flatten().tolist():
                    positions = torch.nonzero(unresolved[row], as_tuple=False).flatten()
                    if positions.numel() == 0:
                        continue
                    best_position = None
                    best_token = None
                    best_confidence = -torch.inf
                    for relative_position in positions.tolist():
                        position = block_start + relative_position
                        position_logits = self.constrain_action_logits(
                            prefix[row, :position].tolist(),
                            logits[row, position] / temperature,
                            target_mass,
                            state=states[row],
                        )
                        probabilities = position_logits.softmax(dim=-1)
                        confidence = probabilities.max(dim=-1).values
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_position = relative_position
                            best_token = int(probabilities.argmax().item())
                    if best_position is None or not torch.isfinite(best_confidence):
                        diagnostics["constraint_dead_ends"] += 1
                        dead_ends = diagnostics["sample_dead_ends"]
                        assert isinstance(dead_ends, list)
                        if len(dead_ends) < 5:
                            dead_ends.append(
                                {
                                    "safe": self._decode_prefix(prefix[row].tolist())[
                                        :512
                                    ],
                                    "heavy_mass": states[row].heavy_mass,
                                    "heavy_atoms": states[row].heavy_atoms,
                                    "valence_sum": states[row].valence_sum,
                                }
                            )
                        active[row] = False
                        unresolved[row] = False
                        continue
                    assert best_token is not None
                    prefix[row, block_start + best_position] = best_token
                    unresolved[row, best_position] = False
                    if best_token == self.eos_token_id:
                        prefix[row, block_start + best_position + 1 :] = (
                            self.mask_token_id
                        )
                        unresolved[row, best_position + 1 :] = False
                    states[row] = self._mass_state(prefix[row].tolist())

            for row in torch.nonzero(active, as_tuple=False).flatten().tolist():
                safe = self._decode_prefix(prefix[row].tolist())
                smiles = self.safe_to_smiles(safe)
                molecule = Chem.MolFromSmiles(smiles) if smiles else None
                is_valid = molecule is not None
                is_mass_valid = self.constraint.accepts_smiles(smiles, target_mass)
                if is_mass_valid or (is_valid and not self.mass_shell_enabled):
                    record_strict_validity(safe)
                    valid += 1
                    results[row] = (safe, smiles, is_mass_valid)
                    active[row] = False
                elif self.eos_token_id in prefix[row, block_start:].tolist():
                    record_strict_validity(safe)
                    diagnostics["eos_terminated"] += 1
                    record_terminal_safe(safe)
                    valid += int(is_valid)
                    active[row] = False

        for row in torch.nonzero(active, as_tuple=False).flatten().tolist():
            safe = self._decode_prefix(prefix[row].tolist())
            smiles = self.safe_to_smiles(safe)
            molecule = Chem.MolFromSmiles(smiles) if smiles else None
            is_valid = molecule is not None
            is_mass_valid = self.constraint.accepts_smiles(smiles, target_mass)
            if is_mass_valid or (is_valid and not self.mass_shell_enabled):
                results[row] = (safe, smiles, is_mass_valid)
            record_strict_validity(safe)
            diagnostics["max_length_terminated"] += 1
            record_terminal_safe(safe)
            valid += int(is_valid)
        return results, valid, diagnostics

    def _canvas_length(self, target_mass: float) -> int:
        content_length = round(target_mass / 8.0)
        content_length = max(content_length, self.model.config.block_width * 2)
        return min(content_length + 2, self.model.config.max_length)

    def _canvas_lengths(
        self,
        target_mass: float,
        candidates: int,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        base = self._canvas_length(target_mass)
        jitter = max(1, min(12, self.model.config.block_width * 2))
        lower = max(3, base - jitter)
        upper = min(self.model.config.max_length, base + jitter)
        if lower == upper:
            return torch.full((candidates,), lower, device=device, dtype=torch.long)
        return torch.randint(
            lower,
            upper + 1,
            (candidates,),
            device=device,
            generator=generator,
        )

    def _decode_canvas_row(self, token_ids: Sequence[int]) -> str:
        compact = []
        for token_id in token_ids:
            if token_id == self.model.config.pad_token_id:
                continue
            compact.append(token_id)
            if token_id == self.eos_token_id:
                break
        return self._decode_prefix(compact)

    def _generate_many_canvas(
        self,
        fingerprint: torch.Tensor,
        target_mass: float,
        *,
        candidates: int,
        diversity_dropout: float,
        temperature: float,
        generator: torch.Generator | None,
    ) -> tuple[list[tuple[str, str, bool] | None], int, dict[str, int | list[str]]]:
        """Generate candidates by filling a fixed masked canvas like DLM sampling."""
        device = next(self.model.parameters()).device
        fingerprint = fingerprint.to(device=device, dtype=torch.float32)
        conditioned = torch.stack(
            [
                perturb_fingerprint(
                    fingerprint, dropout=diversity_dropout, generator=generator
                )
                for _ in range(candidates)
            ]
        ).to(device=device, dtype=torch.float32)
        masses = torch.full((candidates,), target_mass, device=device)
        lengths = self._canvas_lengths(target_mass, candidates, device, generator)
        max_length = int(lengths.max().item())
        canvas = torch.full(
            (candidates, max_length),
            self.model.config.pad_token_id,
            device=device,
            dtype=torch.long,
        )
        canvas[:, 0] = self.bos_token_id
        unresolved = torch.zeros((candidates, max_length), dtype=torch.bool, device=device)
        for row, length in enumerate(lengths.tolist()):
            canvas[row, 1 : length - 1] = self.mask_token_id
            canvas[row, length - 1] = self.eos_token_id
            unresolved[row, 1 : length - 1] = True

        forbidden = set(self.forbidden_token_ids)
        forbidden.add(self.eos_token_id)
        forbidden_ids = sorted(forbidden)
        diagnostics: dict[str, object] = {
            "constraint_dead_ends": 0,
            "eos_terminated": candidates,
            "max_length_terminated": 0,
            "strict_valid": 0,
            "sample_terminal_safes": [],
            "sample_dead_ends": [],
        }

        while unresolved.any():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = self._sampling_logits(canvas, masses, conditioned)
            for row in torch.nonzero(unresolved.any(dim=1), as_tuple=False).flatten().tolist():
                positions = torch.nonzero(unresolved[row], as_tuple=False).flatten()
                if positions.numel() == 0:
                    continue
                block_ids = (positions - 1).div(
                    self.model.config.block_width,
                    rounding_mode="floor",
                )
                positions = positions[block_ids.eq(block_ids.min())]
                probabilities_by_position = []
                confidences = []
                for position in positions.tolist():
                    position_logits = logits[row, position] / temperature
                    if forbidden_ids:
                        position_logits[forbidden_ids] = -torch.inf
                    probabilities = position_logits.softmax(dim=-1)
                    confidence = probabilities.max(dim=-1).values
                    probabilities_by_position.append(probabilities)
                    confidences.append(confidence)
                confidence_tensor = torch.stack(confidences)
                if not torch.isfinite(confidence_tensor).any():
                    diagnostics["constraint_dead_ends"] += 1
                    unresolved[row] = False
                    continue
                selected_index = int(confidence_tensor.argmax().item())
                selected_position = int(positions[selected_index].item())
                token = int(probabilities_by_position[selected_index].argmax().item())
                canvas[row, selected_position] = token
                unresolved[row, selected_position] = False

        results: list[tuple[str, str, bool] | None] = [None] * candidates
        valid = 0
        terminal_examples = diagnostics["sample_terminal_safes"]
        assert isinstance(terminal_examples, list)
        for row in range(candidates):
            safe = self._decode_canvas_row(canvas[row].tolist())
            smiles = self.safe_to_smiles(safe)
            molecule = Chem.MolFromSmiles(smiles) if smiles else None
            is_valid = molecule is not None
            strict_smiles = self.strict_safe_to_smiles(safe)
            strict_molecule = (
                Chem.MolFromSmiles(strict_smiles) if strict_smiles else None
            )
            diagnostics["strict_valid"] += int(strict_molecule is not None)
            is_mass_valid = self.constraint.accepts_smiles(smiles, target_mass)
            valid += int(is_valid)
            if len(terminal_examples) < 5:
                terminal_examples.append(safe[:512])
            if is_mass_valid or (is_valid and not self.mass_shell_enabled):
                results[row] = (safe, smiles, is_mass_valid)
        return results, valid, diagnostics


def _morgan(molecule: Chem.Mol):
    return AllChem.GetMorganGenerator(radius=2, fpSize=4096).GetFingerprint(molecule)


def _fingerprint(array: torch.Tensor):
    bit_vector = DataStructs.ExplicitBitVect(array.numel())
    for index in torch.nonzero(array.reshape(-1), as_tuple=False).flatten().tolist():
        bit_vector.SetBit(index)
    return bit_vector
