"""Prove the lazy grammar probe commits the token the full support would commit.

The block lane used to mask the whole vocabulary at every position, which is the
whole wall clock: 270 ms at prefix length 30 and 5-23 s at 65-88 against ~10 ms
for a model forward. ``MarlinSampler._probe_token`` instead walks the ranked
tokens and stops at the first one ``SafeGrammarMask.admits`` accepts.

This script replays the committed prefixes of a real decode trace, runs the model
that produced it to get real logits, and asserts three things at every position:

1. the probe commits the same token as the full-support path, under both
   ``argmax`` and ``multinomial`` selection and over several draws;
2. the generator ends in the same state, so the RNG stream stays aligned and the
   rest of the decode is unchanged;
3. ``admits`` and the full support agree token for token, which makes the first
   claim hold for every logit vector rather than only the ones observed.

It also times both paths per prefix-length bucket with the mask caches cleared,
because a warm cache is not what a decode sees.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from marlin.grammar import SafeGrammarMask  # noqa: E402
from marlin.mass_shell import MassShellConstraint, MassShellState  # noqa: E402
from marlin.model import MarlinDecoder, MarlinDecoderConfig  # noqa: E402
from marlin.noise import perturb_fingerprint  # noqa: E402
from marlin.sampler import MarlinSampler  # noqa: E402
from marlin.token_properties import (  # noqa: E402
    build_token_property_table,
    foreign_element_token_ids,
    isotope_token_ids,
)
from marlin.tokenizer import load_safe_tokenizer  # noqa: E402


def bucket(length: int) -> str:
    for upper in (16, 32, 48, 64, 80):
        if length < upper:
            return f"<{upper}"
    return ">=80"


def committed_prefixes(paths, bos_token_id):
    """Rebuild every prefix a traced decode actually scored.

    A trace step records the batch, the row, the absolute position and the token
    committed there. The block lane resolves the leftmost unresolved position
    first, so the prefix a step saw is the bos token plus every token committed
    at a smaller position in the same row.
    """
    for path in paths:
        for line in Path(path).read_text().splitlines():
            record = json.loads(line)
            mass = float(record["neutral_mass"])
            committed: dict[tuple[int, int], dict[int, int]] = {}
            for step in record["steps"]:
                if "token" not in step:
                    continue
                key = (int(step["batch"]), int(step["row"]))
                slot = committed.setdefault(key, {})
                position = int(step["position"])
                ordered = [slot[p] for p in range(1, position) if p in slot]
                if len(ordered) == position - 1:
                    yield (
                        record["spec_name"],
                        [bos_token_id] + ordered,
                        mass,
                        int(step["token"]),
                    )
                slot[position] = int(step["token"])


def load_decoder(path: Path) -> MarlinDecoder:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = MarlinDecoderConfig(**checkpoint["hyper_parameters"]["config"])
    state = {
        key.removeprefix("decoder."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("decoder.")
    }
    model = MarlinDecoder(config)
    model.load_state_dict(state, strict=True)
    return model.eval()


def clear_mask_caches(mask: SafeGrammarMask) -> None:
    for cache in (
        mask._mass_reachable_token_ids,
        mask._valid_token_ids,
        mask._support_mask,
        mask._prefix_scan,
        mask._decoded_prefix,
        mask._has_syntactic_completion,
    ):
        cache.cache_clear()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fingerprints", type=Path, required=True)
    parser.add_argument("--fingerprint-key", default="probs")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--diversity-dropout", type=float, default=0.3)
    parser.add_argument("--ppm-tolerance", type=float, default=10.0)
    parser.add_argument("--valence-slack", type=float, default=4.0)
    parser.add_argument("--probe-width", type=int, default=16)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--support-check-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-prefixes", type=int, default=None)
    parser.add_argument("--time-budget-seconds", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    tokenizer = load_safe_tokenizer(arguments.tokenizer)
    special_ids = {
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.mask_token_id,
        tokenizer.pad_token_id,
    }
    token_strings = [
        tokenizer.convert_ids_to_tokens(index) for index in range(len(tokenizer))
    ]
    masses, atoms, valences = build_token_property_table(
        len(tokenizer), tokenizer.convert_ids_to_tokens, special_ids
    )
    constraint = MassShellConstraint(
        masses,
        atoms,
        valences,
        ppm_tolerance=arguments.ppm_tolerance,
        valence_slack=arguments.valence_slack,
        eos_boost=1.0,
        eos_token_id=tokenizer.eos_token_id,
    )
    chemistry_forbidden_ids = tuple(
        sorted(
            set(isotope_token_ids(token_strings))
            | set(foreign_element_token_ids(token_strings))
        )
    )
    mask = SafeGrammarMask(
        token_strings,
        lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
        eos_token_id=tokenizer.eos_token_id,
        mask_token_id=tokenizer.mask_token_id,
        special_token_ids=tuple(special_ids) + (tokenizer.unk_token_id,),
        forbidden_token_ids=chemistry_forbidden_ids,
        ppm_tolerance=arguments.ppm_tolerance,
        valence_slack=arguments.valence_slack,
        mass_reachability_prune=True,
        restrict_organic_elements=True,
        forbid_isotopes=True,
    )
    forbidden = tuple(
        token_id
        for token_id in (
            tokenizer.unk_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.mask_token_id,
            tokenizer.pad_token_id,
        )
        + chemistry_forbidden_ids
        if token_id != tokenizer.eos_token_id
    )

    model = load_decoder(arguments.checkpoint)
    sampler = MarlinSampler(
        model,
        constraint,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        mask_token_id=tokenizer.mask_token_id,
        decode_tokens=lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
        safe_to_smiles=lambda safe: None,
        grammar_mask=mask,
        forbidden_token_ids=forbidden,
        lazy_probe_width=arguments.probe_width,
    )

    with np.load(arguments.fingerprints, allow_pickle=False) as arrays:
        values = np.asarray(arrays[arguments.fingerprint_key])
        positions = {
            str(name): index for index, name in enumerate(arrays["spectrum_ids"])
        }

    seen: set[tuple[tuple[int, ...], float]] = set()
    replayed = []
    for spec, prefix, mass, token in committed_prefixes(
        arguments.trace, tokenizer.bos_token_id
    ):
        key = (tuple(prefix), mass)
        if key in seen:
            continue
        seen.add(key)
        replayed.append((spec, prefix, mass, token))
    # A run stopped by the time budget must still be a fair sample of prefix
    # lengths, and trace order is length order within a row.
    random.Random(arguments.seed).shuffle(replayed)
    if arguments.max_prefixes is not None:
        replayed = replayed[: arguments.max_prefixes]

    conditioning: dict[str, torch.Tensor] = {}
    for spec, _, _, _ in replayed:
        if spec in conditioning:
            continue
        row = torch.tensor(values[positions[spec]], dtype=torch.float32)
        row = torch.where(row >= arguments.threshold, row, torch.zeros_like(row))
        generator = torch.Generator().manual_seed(arguments.seed)
        conditioning[spec] = perturb_fingerprint(
            row, dropout=arguments.diversity_dropout, generator=generator
        ).reshape(1, -1)

    checked = 0
    mismatches = []
    rng_mismatches = 0
    support_prefixes = 0
    support_disagreements = []
    probe_ranks: Counter[int] = Counter()
    fallbacks = 0
    misses = 0
    dead_ends = 0
    timings: dict[str, list[tuple[float, float]]] = defaultdict(list)
    started = time.perf_counter()

    for index, (spec, prefix, mass, _) in enumerate(replayed):
        if (
            arguments.time_budget_seconds is not None
            and time.perf_counter() - started >= arguments.time_budget_seconds
        ):
            print(f"stopping after {index} prefixes: time budget spent", flush=True)
            replayed = replayed[:index]
            break
        input_ids = torch.tensor([prefix + [tokenizer.mask_token_id]])
        with torch.no_grad():
            logits = sampler._sampling_logits(
                input_ids,
                torch.tensor([mass]),
                conditioning[spec],
            )[0, len(prefix)]
        state = MassShellState()
        for token_id in prefix[1:]:
            state = constraint.advance(state, token_id)
        shell = constraint.apply(logits.clone(), state, mass, allow_early_eos=False)
        shell[list(forbidden)] = -torch.inf

        clear_mask_caches(mask)
        full_started = time.perf_counter()
        masked = mask(prefix, shell.clone(), mass)
        full_seconds = time.perf_counter() - full_started
        probabilities = masked.softmax(dim=-1)
        alive = bool(torch.isfinite(probabilities.max(dim=-1).values))

        clear_mask_caches(mask)
        lazy_started = time.perf_counter()
        sampler.sample_tokens = True
        probe_generator = torch.Generator().manual_seed(arguments.seed)
        probed = sampler._probe_token(prefix, shell.clone(), mass, probe_generator)
        lazy_seconds = time.perf_counter() - lazy_started
        timings[bucket(len(prefix))].append((full_seconds, lazy_seconds))

        # Every token, so the identity holds for any logit vector and not only
        # for the ones this checkpoint happens to produce. It costs a second
        # vocabulary walk, so it is sampled rather than run at every prefix.
        if index % arguments.support_check_every == 0:
            support = set(torch.nonzero(torch.isfinite(masked)).flatten().tolist())
            admitted = {
                token_id
                for token_id in range(len(token_strings))
                if mask.admits(prefix, token_id, mass)
            }
            support_prefixes += 1
            if admitted != support:
                support_disagreements.append(
                    {
                        "spec": spec,
                        "prefix_length": len(prefix),
                        "only_admits": sorted(admitted - support)[:8],
                        "only_support": sorted(support - admitted)[:8],
                    }
                )

        for mode in (False, True):
            sampler.sample_tokens = mode
            draws = arguments.draws if mode else 1
            for draw in range(draws):
                seed = arguments.seed + 1000 * draw
                full_generator = torch.Generator().manual_seed(seed)
                if not alive:
                    expected = None
                elif mode:
                    expected = int(
                        torch.multinomial(
                            probabilities, num_samples=1, generator=full_generator
                        ).item()
                    )
                else:
                    expected = int(probabilities.argmax().item())
                lazy_generator = torch.Generator().manual_seed(seed)
                before_fallbacks = sampler.lazy_probe_fallbacks
                before_misses = sampler.lazy_probe_misses
                before_calls = sampler.lazy_probe_admits_calls
                actual = sampler._probe_token(
                    prefix, shell.clone(), mass, lazy_generator
                )
                checked += 1
                probe_ranks[sampler.lazy_probe_admits_calls - before_calls] += 1
                if sampler.lazy_probe_fallbacks > before_fallbacks:
                    fallbacks += 1
                if sampler.lazy_probe_misses > before_misses:
                    misses += 1
                if actual is None:
                    dead_ends += 1
                if actual != expected:
                    mismatches.append(
                        {
                            "spec": spec,
                            "prefix_length": len(prefix),
                            "sample_tokens": mode,
                            "seed": seed,
                            "expected": expected,
                            "actual": actual,
                        }
                    )
                if not torch.equal(
                    full_generator.get_state(), lazy_generator.get_state()
                ):
                    rng_mismatches += 1
        if (index + 1) % 25 == 0:
            print(
                f"{index + 1}/{len(replayed)} prefixes, {checked} positions, "
                f"{len(mismatches)} mismatches, {misses} width misses, "
                f"{fallbacks} full-support fallbacks",
                flush=True,
            )

    report = {
        "prefixes": len(replayed),
        "positions_checked": checked,
        "mismatches": len(mismatches),
        "mismatch_examples": mismatches[:20],
        "rng_state_mismatches": rng_mismatches,
        "dead_ends": dead_ends,
        "support_prefixes_compared": support_prefixes,
        "support_disagreements": len(support_disagreements),
        "support_disagreement_examples": support_disagreements[:20],
        "probe_width_misses": misses,
        "probe_width_miss_rate": misses / checked if checked else 0.0,
        "full_support_fallbacks": fallbacks,
        "full_support_fallback_rate": fallbacks / checked if checked else 0.0,
        "probe_admits_calls_histogram": dict(sorted(probe_ranks.items())),
        "probe_width": arguments.probe_width,
        "timings": {
            name: {
                "prefixes": len(pairs),
                "median_full_ms": 1000 * statistics.median(f for f, _ in pairs),
                "median_lazy_ms": 1000 * statistics.median(l for _, l in pairs),
                "median_speedup": statistics.median(
                    f / l for f, l in pairs if l > 0
                ),
                "total_full_s": sum(f for f, _ in pairs),
                "total_lazy_s": sum(l for _, l in pairs),
            }
            for name, pairs in sorted(timings.items())
        },
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "timings"}, indent=2))
    print(json.dumps(report["timings"], indent=2))
    return 1 if mismatches or rng_mismatches or support_disagreements else 0


if __name__ == "__main__":
    raise SystemExit(main())
