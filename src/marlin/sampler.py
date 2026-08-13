"""Mass-shell constrained block-diffusion sampling and ranking."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

from marlin.mass_shell import MassShellConstraint, MassShellState, conditioning_mass
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
    constraint_dead_ends: int
    eos_terminated: int
    max_length_terminated: int
    sample_terminal_safes: tuple[str, ...]
    sample_dead_ends: tuple[dict[str, float | int | str], ...]
    # True when a time budget stopped generation before the requested budget was
    # spent. ``attempts`` then reports what was actually tried, so a truncated
    # spectrum can never be mistaken for a fully searched one.
    truncated: bool = False


# A committed token stays the same when the ranking is rescaled by the softmax
# normaliser the grammar support would have supplied, but float32 only carries
# about seven digits and ``exp(x - max)`` loses a few of them at large offsets.
# A winner this close to the next-ranked token is therefore decided by the full
# support instead of by the probe.
_PROBE_MARGIN = 1e-4
# ``torch.isfinite`` on a scalar stands in for the confidence of a position the
# probe resolved: the block loop only reads that value to reject a dead end, and
# with a grammar mask exactly one position is ever scored.
_PROBE_RESOLVED = torch.tensor(1.0)


def _capture_rng(
    generator: torch.Generator | None, device: torch.device
) -> tuple[torch.Generator | None, torch.device, torch.Tensor]:
    """Record the draw state the probe is about to advance."""
    if generator is not None:
        return generator, device, generator.get_state()
    if device.type == "cuda":
        return None, device, torch.cuda.get_rng_state(device)
    return None, device, torch.random.get_rng_state()


def _restore_rng(
    state: tuple[torch.Generator | None, torch.device, torch.Tensor],
) -> None:
    """Undo a draw the full-support path would never have made."""
    generator, device, saved = state
    if generator is not None:
        generator.set_state(saved)
    elif device.type == "cuda":
        torch.cuda.set_rng_state(saved, device)
    else:
        torch.random.set_rng_state(saved)


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
        grammar_mask: Callable[
            [Sequence[int], torch.Tensor, float | None], torch.Tensor
        ]
        | None = None,
        forbidden_token_ids: Sequence[int] = (),
        mass_shell_enabled: bool = True,
        generation_mode: str = "block",
        sample_tokens: bool = False,
        trace: list[dict[str, object]] | None = None,
        lazy_probe_width: int = 16,
    ) -> None:
        if generation_mode not in {"block", "canvas"}:
            raise ValueError("generation_mode must be 'block' or 'canvas'")
        if lazy_probe_width < 0:
            raise ValueError("lazy_probe_width must not be negative")
        self.model = model
        self.constraint = constraint
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.mask_token_id = mask_token_id
        self.decode_tokens = decode_tokens
        self.safe_to_smiles = safe_to_smiles
        self.grammar_mask = grammar_mask
        self.forbidden_token_ids = tuple(forbidden_token_ids)
        self.mass_shell_enabled = mass_shell_enabled
        self.generation_mode = generation_mode
        self.sample_tokens = sample_tokens
        # Passing a list makes the block path record every committed token and
        # every ending, so a run can be replayed attempt by attempt. Left None,
        # the decoding path is untouched.
        self.trace = trace
        self._trace_batch = 0
        # Probing the ranked tokens one at a time instead of masking the whole
        # vocabulary. Set to 0 to force the full-support path.
        self.lazy_probe_width = lazy_probe_width
        self.lazy_probe_positions = 0
        # Positions whose answer was not among the first ``lazy_probe_width``
        # ranked tokens, and positions that had to build the support anyway.
        self.lazy_probe_misses = 0
        self.lazy_probe_fallbacks = 0
        self.lazy_probe_admits_calls = 0

    @property
    def _lazy_probe_available(self) -> bool:
        """Report whether the block lane may resolve a position by probing.

        The probe needs a mask that can answer about one token
        (``SafeGrammarMask.admits``); it cannot serve a trace, which records the
        support size and the four most probable tokens and therefore needs the
        whole masked distribution anyway.
        """
        return (
            self.lazy_probe_width > 0
            and self.trace is None
            and self.grammar_mask is not None
            and self.mask_token_id is not None
            and callable(getattr(self.grammar_mask, "admits", None))
        )

    def _probe_token(
        self,
        prefix_ids: list[int],
        shell_logits: torch.Tensor,
        target_mass: float,
        generator: torch.Generator | None,
    ) -> int | None:
        """Commit the token the full-support path would commit, without building it.

        The full-support call is the whole wall clock: 270 ms at prefix length 30
        and 5-23 s at 65-88, against 0.47 ms for a single-token ``admits`` probe
        and ~10 ms for the model forward.

        Both selection rules read the masked distribution only through an argmax.
        ``probabilities.argmax()`` is the highest-scoring token the grammar
        admits, and ``torch.multinomial(probabilities, 1)`` is implemented as
        ``argmax(probabilities / q)`` with ``q ~ Exp(1)`` drawn over the whole
        vocabulary independently of ``probabilities`` -- one ``multinomial_out``
        serves both CPU and CUDA in ``native_functions.yaml``, and the draw it
        makes is the ``exponential_`` call reproduced here. Restricting the
        support only removes candidates and rescales the rest by one positive
        constant, so in both cases the answer is the first token the grammar
        admits when the vocabulary is walked in the order the mass shell alone
        already fixes.

        ``lazy_probe_width`` is where a position stops being cheap, not where the
        walk stops: the positions whose answer is ranked low are exactly the long
        prefixes whose support costs 13-23 s to build, and walking on stops at
        the answer instead of testing every token past it. The support is built
        only when the ranking cannot be trusted -- a score that underflowed to
        zero, or a winner too close to the next token to survive the float32
        rounding of the path being reproduced.

        Returns the token id, or ``None`` for a dead end. On a dead end the
        generator is left exactly where the full-support path would have left it,
        because that path never reaches its ``multinomial`` call.
        """
        assert self.grammar_mask is not None
        if not torch.isfinite(shell_logits).any():
            # The full-support path softmaxes to NaN here and never draws.
            return None
        self.lazy_probe_positions += 1
        finite = torch.isfinite(shell_logits)
        state = None
        quantiles = None
        if self.sample_tokens:
            state = _capture_rng(generator, shell_logits.device)
            # Drawn exactly as ``torch.multinomial`` draws it, from the same
            # generator, so the stream advances identically either way.
            quantiles = torch.empty_like(shell_logits).exponential_(
                1, generator=generator
            )
        # float64 keeps the ranking free of the rounding the float32 path carries,
        # so ``_PROBE_MARGIN`` is a bound on that path's error and not on ours.
        scores = shell_logits.to(torch.float64).softmax(dim=-1)
        if quantiles is not None:
            scores = scores / quantiles.to(torch.float64)
        # A token the mass shell already rejected must never win, and 0/0 from a
        # zero quantile would otherwise sort ahead of everything as NaN.
        scores = torch.where(finite, scores, torch.full_like(scores, -1.0))
        candidates = int(finite.sum().item())
        values, ranked = scores.sort(descending=True, stable=True)
        values = values[:candidates].tolist()
        trusted = True
        for index, token_id in enumerate(ranked[:candidates].tolist()):
            if not values[index] > 0.0:
                # Underflow lost the ranking; let the full support decide.
                trusted = False
                break
            if index == self.lazy_probe_width:
                self.lazy_probe_misses += 1
            self.lazy_probe_admits_calls += 1
            if not self.grammar_mask.admits(prefix_ids, token_id, target_mass):
                continue
            runner_up = values[index + 1] if index + 1 < candidates else 0.0
            if runner_up > 0.0 and values[index] <= runner_up * (1.0 + _PROBE_MARGIN):
                trusted = False
                break
            return token_id
        if trusted:
            # Every token the mass shell left standing was tested and refused, so
            # the masked distribution is all -inf and the position is a dead end.
            if state is not None:
                _restore_rng(state)
            return None
        self.lazy_probe_fallbacks += 1
        masked = self.grammar_mask(prefix_ids, shell_logits, target_mass)
        probabilities = masked.softmax(dim=-1)
        if not torch.isfinite(probabilities.max(dim=-1).values):
            if state is not None:
                _restore_rng(state)
            return None
        if quantiles is not None:
            return int((probabilities / quantiles).argmax().item())
        return int(probabilities.argmax().item())

    def _sampling_logits(
        self,
        input_ids: torch.Tensor,
        precursor_mass: torch.Tensor,
        fingerprint: torch.Tensor,
        isotope_ratios: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sampling_logits = getattr(self.model, "sampling_logits", None)
        forward = (
            sampling_logits
            if sampling_logits is not None
            else lambda *arguments: self.model(*arguments)
        )
        if isotope_ratios is None:
            return forward(input_ids, precursor_mass, fingerprint)
        return forward(input_ids, precursor_mass, fingerprint, isotope_ratios)

    def _isotope_batch(
        self, isotope_ratios: Sequence[float] | None, rows: int, device: torch.device
    ) -> torch.Tensor | None:
        """Expand a per-spectrum isotope pair to one conditioning token per row.

        Training always emits this token, so omitting it at sampling time moves
        every forward pass off the training conditioning layout.
        """
        if isotope_ratios is None:
            return None
        values = tuple(float(value) for value in isotope_ratios)
        if len(values) != 2:
            raise ValueError("isotope_ratios must hold M+1/M and M+2/M")
        return torch.tensor(values, device=device, dtype=torch.float32).expand(rows, 2)

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
                    position_logits = logits[position] / temperature
                    if self.mass_shell_enabled:
                        position_logits = self.constraint.apply(
                            position_logits,
                            state,
                            target_mass,
                            allow_early_eos=self.mask_token_id
                            in prefix[:position],
                        )
                    if self.forbidden_token_ids:
                        position_logits[list(self.forbidden_token_ids)] = -torch.inf
                    if self.grammar_mask is not None:
                        position_logits = self.grammar_mask(
                            prefix[:position], position_logits, target_mass
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
        candidate_batch_size: int | None = None,
    ) -> list[MarlinCandidate]:
        ranked, _ = self.generate_ranked_with_stats(
            fingerprint,
            target_mass,
            candidates=candidates,
            diversity_dropout=diversity_dropout,
            temperature=temperature,
            generator=generator,
            candidate_batch_size=candidate_batch_size,
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
        candidate_batch_size: int | None = None,
        isotope_ratios: Sequence[float] | None = None,
        time_budget_seconds: float | None = None,
    ) -> tuple[list[MarlinCandidate], MarlinGenerationStats]:
        if candidates <= 0:
            raise ValueError("candidates must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if candidate_batch_size is not None and candidate_batch_size <= 0:
            raise ValueError("candidate_batch_size must be positive")
        if time_budget_seconds is not None and time_budget_seconds <= 0:
            raise ValueError("time_budget_seconds must be positive")
        original = fingerprint.to(torch.float32)
        generator_fn = (
            self._generate_many_canvas
            if self.generation_mode == "canvas"
            else self._generate_many
        )
        batch_size = min(candidate_batch_size or candidates, candidates)
        generated: list[tuple[str, str, bool] | None] = []
        valid = 0
        diagnostics: dict[str, int | list] = {
            "constraint_dead_ends": 0,
            "eos_terminated": 0,
            "max_length_terminated": 0,
            "sample_terminal_safes": [],
            "sample_dead_ends": [],
        }
        attempted = 0
        truncated = False
        deadline = (
            None if time_budget_seconds is None
            else time.perf_counter() + time_budget_seconds
        )
        for start in range(0, candidates, batch_size):
            # Checked at batch boundaries only, so no candidate is ever half
            # generated and the RNG stream stays exactly as it would have been.
            if deadline is not None and start > 0 and time.perf_counter() >= deadline:
                truncated = True
                break
            batch_candidates = min(batch_size, candidates - start)
            attempted += batch_candidates
            batch_generated, batch_valid, batch_diagnostics = generator_fn(
                original,
                target_mass,
                candidates=batch_candidates,
                diversity_dropout=diversity_dropout,
                temperature=temperature,
                generator=generator,
                isotope_ratios=isotope_ratios,
                deadline=deadline,
            )
            if deadline is not None and time.perf_counter() >= deadline:
                truncated = True
            generated.extend(batch_generated)
            valid += batch_valid
            for name in (
                "constraint_dead_ends",
                "eos_terminated",
                "max_length_terminated",
            ):
                diagnostics[name] = int(diagnostics[name]) + int(
                    batch_diagnostics[name]
                )
            for name in ("sample_terminal_safes", "sample_dead_ends"):
                samples = diagnostics[name]
                assert isinstance(samples, list)
                batch_samples = batch_diagnostics[name]
                assert isinstance(batch_samples, list)
                samples.extend(batch_samples[: 5 - len(samples)])
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

        reference = _fingerprint((original > 0.5).to(torch.float32))
        ranked = []
        unique_mass_valid = 0
        for safe, smiles, is_mass_valid in unique.values():
            unique_mass_valid += int(is_mass_valid)
            molecule = Chem.MolFromSmiles(smiles)
            # The reported error has to be the error the acceptance test measures,
            # or a charged candidate reads 1,300 ppm off while being accepted.
            candidate_mass = conditioning_mass(molecule)
            ranked.append(
                MarlinCandidate(
                    smiles=smiles,
                    safe=safe,
                    tanimoto=DataStructs.TanimotoSimilarity(
                        reference, _morgan(molecule)
                    ),
                    mass_error_ppm=1e6 * (candidate_mass - target_mass) / target_mass,
                )
            )
        ranked = sorted(
            ranked,
            key=lambda candidate: (-candidate.tanimoto, abs(candidate.mass_error_ppm)),
        )
        return ranked, MarlinGenerationStats(
            attempts=attempted,
            valid=valid,
            mass_valid=mass_valid,
            unique_mass_valid=unique_mass_valid,
            constraint_dead_ends=diagnostics["constraint_dead_ends"],
            eos_terminated=diagnostics["eos_terminated"],
            max_length_terminated=diagnostics["max_length_terminated"],
            sample_terminal_safes=tuple(diagnostics["sample_terminal_safes"]),
            sample_dead_ends=tuple(diagnostics["sample_dead_ends"]),
            truncated=truncated,
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
        isotope_ratios: Sequence[float] | None = None,
        deadline: float | None = None,
    ) -> tuple[list[tuple[str, str, bool] | None], int, dict[str, int | list[str]]]:
        """Generate candidates in one GPU batch and retain per-row constraints."""
        device = next(self.model.parameters()).device
        isotopes = self._isotope_batch(isotope_ratios, candidates, device)
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
        batch = self._trace_batch
        self._trace_batch += 1
        block_index = -1
        diagnostics: dict[str, object] = {
            "constraint_dead_ends": 0,
            "eos_terminated": 0,
            "max_length_terminated": 0,
            "sample_terminal_safes": [],
            "sample_dead_ends": [],
        }

        def record_terminal_safe(safe: str) -> None:
            examples = diagnostics["sample_terminal_safes"]
            assert isinstance(examples, list)
            if len(examples) < 5:
                examples.append(safe[:512])

        def expired() -> bool:
            return deadline is not None and time.perf_counter() >= deadline

        lazy_probe = self._lazy_probe_available
        while prefix.shape[1] < self.model.config.max_length:
            if not active.any():
                break
            if expired():
                # A batch holding every candidate never reaches a batch boundary,
                # so a deadline read only there is never read: one spectrum of the
                # clean panel ran 19,311 s against a 1,800 s budget. The rows still
                # decoding are abandoned unfinished, exactly as the length cap
                # abandons them.
                break
            block_start = prefix.shape[1]
            block_width = self._next_block_width(block_start)
            block_index += 1
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
                if expired():
                    # A block is 8 positions and one of them could take minutes,
                    # so a deadline read only between blocks overshoots: a 900 s
                    # budget produced a 1,308 s spectrum. The positions left in
                    # this block stay masked and decode away, exactly as the
                    # length cap leaves them.
                    break
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    logits = self._sampling_logits(
                        prefix, masses, conditioned, isotopes
                    )
                for row in torch.nonzero(active, as_tuple=False).flatten().tolist():
                    positions = torch.nonzero(unresolved[row], as_tuple=False).flatten()
                    if positions.numel() == 0:
                        continue
                    if expired():
                        # One row's fallback to the full support can still cost
                        # seconds, so the budget is read per row as well.
                        break
                    best_position = None
                    best_token = None
                    best_confidence = -torch.inf
                    best_support = 0
                    best_alternatives: list[tuple[int, float]] = []
                    candidate_positions = positions.tolist()
                    if self.grammar_mask is not None and self.mask_token_id is not None:
                        # Every position except the leftmost unresolved one still
                        # has a hole behind it, and the grammar mask gives such a
                        # position no support at all, so its softmax is NaN and it
                        # can never win the confidence comparison. Scoring it costs
                        # a mass-shell pass, a forbidden-token scatter and a full
                        # vocabulary walk for a result that is discarded.
                        candidate_positions = candidate_positions[:1]
                    for relative_position in candidate_positions:
                        position = block_start + relative_position
                        position_logits = logits[row, position] / temperature
                        if self.mass_shell_enabled:
                            position_logits = self.constraint.apply(
                                position_logits,
                                states[row],
                                target_mass,
                                allow_early_eos=self.mask_token_id
                                in prefix[row, :position],
                            )
                        if self.forbidden_token_ids:
                            position_logits[list(self.forbidden_token_ids)] = -torch.inf
                        if lazy_probe:
                            # The probe writes ``best_*`` without comparing
                            # confidences, which is only sound because it is
                            # reached once: ``_lazy_probe_available`` demands the
                            # same grammar mask and mask token that truncated
                            # ``candidate_positions`` to the leftmost hole above.
                            assert len(candidate_positions) == 1
                            probed = self._probe_token(
                                prefix[row, :position].tolist(),
                                position_logits,
                                target_mass,
                                generator,
                            )
                            if probed is None:
                                continue
                            best_confidence = _PROBE_RESOLVED
                            best_position = relative_position
                            best_token = probed
                            continue
                        if self.grammar_mask is not None:
                            position_logits = self.grammar_mask(
                                prefix[row, :position].tolist(),
                                position_logits,
                                target_mass,
                            )
                        probabilities = position_logits.softmax(dim=-1)
                        confidence = probabilities.max(dim=-1).values
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_position = relative_position
                            if self.trace is not None:
                                best_support = int(
                                    torch.isfinite(position_logits).sum().item()
                                )
                                ranked = probabilities.topk(
                                    min(4, probabilities.numel())
                                )
                                best_alternatives = [
                                    (int(token), float(probability))
                                    for token, probability in zip(
                                        ranked.indices.tolist(), ranked.values.tolist()
                                    )
                                ]
                            best_token = int(
                                torch.multinomial(
                                    probabilities,
                                    num_samples=1,
                                    generator=generator,
                                ).item()
                                if self.sample_tokens
                                else probabilities.argmax().item()
                            )
                    if best_position is None or not torch.isfinite(best_confidence):
                        diagnostics["constraint_dead_ends"] += 1
                        if self.trace is not None:
                            self.trace.append(
                                {
                                    "batch": batch,
                                    "row": row,
                                    "block": block_index,
                                    "position": block_start
                                    + (
                                        int(positions[0].item())
                                        if positions.numel()
                                        else 0
                                    ),
                                    "ending": "dead_end",
                                    "safe": self._decode_prefix(prefix[row].tolist())[
                                        :512
                                    ],
                                    "heavy_mass": states[row].heavy_mass,
                                    "heavy_atoms": states[row].heavy_atoms,
                                }
                            )
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
                    if self.trace is not None:
                        self.trace.append(
                            {
                                "batch": batch,
                                "row": row,
                                "block": block_index,
                                "position": block_start + best_position,
                                "token": int(best_token),
                                "confidence": float(best_confidence),
                                "support": best_support,
                                "alternatives": best_alternatives,
                                "heavy_mass": states[row].heavy_mass,
                                "heavy_atoms": states[row].heavy_atoms,
                            }
                        )
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
                    valid += 1
                    results[row] = (safe, smiles, is_mass_valid)
                    active[row] = False
                    if self.trace is not None:
                        self.trace.append(
                            {
                                "batch": batch,
                                "row": row,
                                "block": block_index,
                                "ending": "accepted",
                                "safe": safe,
                                "smiles": smiles,
                            }
                        )
                elif self.eos_token_id in prefix[row, block_start:].tolist():
                    diagnostics["eos_terminated"] += 1
                    if self.trace is not None:
                        self.trace.append(
                            {
                                "batch": batch,
                                "row": row,
                                "block": block_index,
                                "ending": "eos_rejected",
                                "safe": safe,
                                "smiles": smiles,
                                "parsed": bool(is_valid),
                            }
                        )
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
            diagnostics["max_length_terminated"] += 1
            if self.trace is not None:
                self.trace.append(
                    {
                        "batch": batch,
                        "row": row,
                        "block": block_index,
                        "ending": "max_length",
                        "safe": safe,
                        "smiles": smiles,
                        "parsed": bool(is_valid),
                    }
                )
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
        isotope_ratios: Sequence[float] | None = None,
        deadline: float | None = None,
    ) -> tuple[list[tuple[str, str, bool] | None], int, dict[str, int | list[str]]]:
        """Generate candidates by filling a fixed masked canvas like DLM sampling."""
        device = next(self.model.parameters()).device
        isotopes = self._isotope_batch(isotope_ratios, candidates, device)
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
            "sample_terminal_safes": [],
            "sample_dead_ends": [],
        }

        while unresolved.any():
            if deadline is not None and time.perf_counter() >= deadline:
                break
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits = self._sampling_logits(canvas, masses, conditioned, isotopes)
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
                selected_probabilities = probabilities_by_position[selected_index]
                token = int(
                    torch.multinomial(
                        selected_probabilities,
                        num_samples=1,
                        generator=generator,
                    ).item()
                    if self.sample_tokens
                    else selected_probabilities.argmax().item()
                )
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
