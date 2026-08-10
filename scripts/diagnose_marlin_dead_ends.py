#!/usr/bin/env python3
"""Diagnose the constraint dead ends and parse failures of a MARLIN evaluation.

The script reads a ``predictions.jsonl`` written by
``scripts/evaluate_marlin_nplib1.py`` and answers four questions with counts:

1. which target properties separate the spectra whose every attempt ended in a
   constraint dead end from the spectra that returned a candidate;
2. whether each recorded dead-end prefix has an empty *syntax* support (a
   grammar deadlock) or an empty *mass-reachable* support (a mass-shell
   deadlock), asked of the same :class:`~marlin.grammar.SafeGrammarMask` the run
   used;
3. how far along the mass budget the mass-shell deadlocks died;
4. why the terminal SAFE strings of the spectra that completed without a parsed
   molecule fail in RDKit.

Everything is written to one JSON report; nothing in the decoding path is
touched.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import time
from ast import literal_eval
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from dlm.utils.utils_chem import safe_to_smiles, smiles_to_safe
from marlin.grammar import SafeGrammarMask, _scan
from marlin.mass_shell import MassShellConstraint
from marlin.token_properties import (
    build_token_property_table,
    foreign_element_token_ids,
    isotope_token_ids,
)
from marlin.tokenizer import load_safe_tokenizer

RDLogger.DisableLog("rdApp.*")

PARSE_BUCKETS = (
    ("unclosed_ring", re.compile(r"unclosed ring")),
    ("kekulization", re.compile(r"[Cc]an't kekulize")),
    ("valence", re.compile(r"greater than permitted")),
    ("duplicate_ring_bond", re.compile(r"duplicates? bond|[Rr]ing closure")),
    ("aromatic_non_ring", re.compile(r"non-ring atom .* marked aromatic")),
    ("syntax", re.compile(r"syntax error|extra (open|close)")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="runtime-inputs directory holding tokenizer.json and test/",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--fingerprint-key", default="probs")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--ppm-tolerance", type=float, default=10.0)
    parser.add_argument("--valence-slack", type=float, default=4.0)
    parser.add_argument(
        "--dead-end-scope",
        choices=("all_dead_end", "all"),
        default="all_dead_end",
        help="probe the prefixes of the all-dead-end spectra only, or of every "
        "spectrum that recorded one",
    )
    parser.add_argument(
        "--gold-vocabulary-splits",
        default="train,val,test",
        help="splits whose gold answers are checked against the vocabulary",
    )
    parser.add_argument(
        "--gold-walk-sample",
        type=int,
        default=0,
        help="replay the gold token sequence through the mask for this many of the "
        "lightest all-dead-end targets, plus every charged one",
    )
    parser.add_argument(
        "--funnel-sample",
        type=int,
        default=0,
        help="re-ask the support of this many dead ends at the point where their "
        "bracket-hydrogen tail begins",
    )
    parser.add_argument(
        "--reuse-dead-ends",
        type=Path,
        default=None,
        help="load an earlier <output>.dead_ends.csv instead of asking the mask "
        "again; the mask probe is the only expensive step here",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def group_of(row: dict) -> str:
    if row["candidate_returned"]:
        return "returned"
    if row["constraint_dead_ends"] == row["attempts"]:
        return "all_dead_end"
    if row["valid"] > 0:
        return "mass_miss"
    return "no_parse"


def target_features(
    rows: list[dict], cache_dir: Path, split: str, key: str, threshold: float, tokenizer
) -> pd.DataFrame:
    metadata_path = cache_dir / split / "metadata.csv"
    fingerprint_path = cache_dir / split / "dreams_predictions.npz"
    metadata = pd.read_csv(metadata_path).set_index("spec_name")
    mismatched = [
        row["spec_name"]
        for row in rows
        if metadata.loc[row["spec_name"], "smiles"] != row["target_smiles"]
    ]
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} predictions disagree with {metadata_path} on the "
            f"target SMILES, first {mismatched[0]}"
        )
    bundle = np.load(fingerprint_path, allow_pickle=False)
    probabilities = bundle[key]
    positions = {str(name): index for index, name in enumerate(bundle["spectrum_ids"])}
    generator = AllChem.GetMorganGenerator(radius=2, fpSize=probabilities.shape[1])
    hydrogen_mass = 1.00782503223
    records = []
    for row in rows:
        molecule = Chem.MolFromSmiles(row["target_smiles"])
        truth = np.asarray(generator.GetFingerprint(molecule), dtype=bool)
        predicted = probabilities[positions[row["spec_name"]]] >= threshold
        intersection = int((predicted & truth).sum())
        union = int((predicted | truth).sum())
        safe = smiles_to_safe(row["target_smiles"])
        token_ids = tokenizer.encode(safe, add_special_tokens=True)
        heavy_atoms = molecule.GetNumHeavyAtoms()
        hydrogens = Chem.AddHs(molecule).GetNumAtoms() - heavy_atoms
        heteroatoms = sum(
            1 for atom in molecule.GetAtoms() if atom.GetSymbol() not in ("C", "H")
        )
        nitrogens = sum(1 for atom in molecule.GetAtoms() if atom.GetSymbol() == "N")
        oxygens = sum(1 for atom in molecule.GetAtoms() if atom.GetSymbol() == "O")
        exact_mass = Descriptors.ExactMolWt(molecule)
        records.append(
            {
                "spec_name": row["spec_name"],
                "group": group_of(row),
                "neutral_mass": row["neutral_mass"],
                "heavy_atoms": heavy_atoms,
                "rings": rdMolDescriptors.CalcNumRings(molecule),
                "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(molecule),
                "hetero_fraction": heteroatoms / heavy_atoms,
                "nitrogen_fraction": nitrogens / heavy_atoms,
                "oxygen_fraction": oxygens / heavy_atoms,
                "nitrogens": nitrogens,
                "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(molecule),
                "safe_fragments": safe.count(".") + 1,
                "gold_tokens": len(token_ids),
                "hydrogen_mass_fraction": hydrogens * hydrogen_mass / exact_mass,
                "formal_charge": Chem.GetFormalCharge(molecule),
                "gold_exact_mass": exact_mass,
                "gold_mass_error_ppm": (exact_mass - row["neutral_mass"])
                / row["neutral_mass"]
                * 1e6,
                "fingerprint_tanimoto": intersection / union if union else float("nan"),
                "fingerprint_recall": intersection / int(truth.sum()),
                "predicted_bits": int(predicted.sum()),
                "true_bits": int(truth.sum()),
                "dead_ends": row["constraint_dead_ends"],
                "valid": row["valid"],
                "mass_valid": row["mass_valid"],
                "eos_terminated": row["eos_terminated"],
                "max_length_terminated": row["max_length_terminated"],
            }
        )
    return pd.DataFrame(records)


def separation_table(
    features: pd.DataFrame,
    columns: list[str],
    left_group: str = "all_dead_end",
    right_group: str = "returned",
) -> list[dict]:
    """Rank features by how well each orders one group against another.

    The statistic is the Mann-Whitney U scaled to an AUC: 0.5 means the two
    groups are interleaved, 0 and 1 mean they are separated.
    """
    from scipy.stats import mannwhitneyu

    table = []
    for column in columns:
        left = features[features.group == left_group][column].dropna()
        right = features[features.group == right_group][column].dropna()
        statistic, pvalue = mannwhitneyu(left, right, alternative="two-sided")
        auc = statistic / (len(left) * len(right))
        table.append(
            {
                "feature": column,
                "left_group": left_group,
                "right_group": right_group,
                "n_left": len(left),
                "n_right": len(right),
                "median_left": float(np.median(left)),
                "median_right": float(np.median(right)),
                "auc_left_over_right": round(float(auc), 3),
                "separation": round(abs(float(auc) - 0.5) * 2, 3),
                "p_value": float(pvalue),
            }
        )
    return sorted(table, key=lambda entry: -entry["separation"])


def _tertiles(values: pd.Series) -> tuple[pd.Series, list[float]]:
    """Split into three near-equal groups, tolerating heavy ties."""
    ranks = values.rank(method="first")
    bins = pd.qcut(ranks, 3, labels=("low", "mid", "high"))
    edges = [float(values[bins == label].max()) for label in ("low", "mid", "high")]
    return bins, edges


def stratified_table(features: pd.DataFrame, row_key: str, column_key: str) -> dict:
    subset = features[features.group.isin(("all_dead_end", "returned"))].copy()
    subset["dead"] = (subset.group == "all_dead_end").astype(int)
    subset["row_bin"], row_edges = _tertiles(subset[row_key])
    subset["column_bin"], column_edges = _tertiles(subset[column_key])
    shares = subset.pivot_table(
        index="row_bin", columns="column_bin", values="dead", aggfunc="mean"
    )
    sizes = subset.pivot_table(
        index="row_bin", columns="column_bin", values="dead", aggfunc="size"
    )
    return {
        "row_key": row_key,
        "column_key": column_key,
        "dead_share": shares.round(3).to_dict(),
        "cell_size": sizes.to_dict(),
        "row_upper_edges": row_edges,
        "column_upper_edges": column_edges,
    }


def build_mask(cache_dir: Path, ppm_tolerance: float, valence_slack: float):
    tokenizer = load_safe_tokenizer(cache_dir / "tokenizer.json")
    token_strings = [
        tokenizer.convert_ids_to_tokens(index) for index in range(len(tokenizer))
    ]
    special_ids = (
        tokenizer.unk_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.mask_token_id,
    )
    forbidden = tuple(
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
        special_token_ids=special_ids,
        forbidden_token_ids=forbidden,
        ppm_tolerance=ppm_tolerance,
        valence_slack=valence_slack,
        mass_reachability_prune=True,
    )
    masses, atoms, valences = build_token_property_table(
        len(tokenizer), tokenizer.convert_ids_to_tokens, special_ids
    )
    constraint = MassShellConstraint(
        masses,
        atoms,
        valences,
        ppm_tolerance=ppm_tolerance,
        valence_slack=valence_slack,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer, mask, constraint, forbidden


ORGANIC = frozenset({"C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I"})


def phantom_atoms(state) -> dict:
    """Count the atoms the grammar's own mass model treats as weightless.

    ``marlin.grammar._ATOM_MASSES`` has no hydrogen entry and no entry for any
    element outside its 18-element table, so ``_advance`` files those atoms at
    mass 0.0. They still consume valence and still count in the mass-shell token
    table, so a prefix holding them means the two mass accountings disagree.
    """
    symbols = [state.atom_symbols[index] for index in sorted(state.atom_symbols)]
    zero_mass = [
        state.atom_symbols[index]
        for index, mass in state.atom_masses.items()
        if mass == 0.0
    ]
    return {
        "atoms": len(symbols),
        "zero_mass_atoms": len(zero_mass),
        "bracket_hydrogen_atoms": sum(1 for symbol in zero_mass if symbol == "H"),
        "foreign_element_atoms": sum(
            1 for symbol in zero_mass if symbol.rstrip("+").capitalize() not in ORGANIC
        ),
        "zero_mass_symbols": sorted(set(zero_mass)),
    }


def probe_dead_ends(
    rows: list[dict],
    features: pd.DataFrame,
    mask,
    constraint,
    forbidden: tuple[int, ...],
    valence_slack: float,
    scope: str,
) -> list[dict]:
    gold_atoms = dict(zip(features.spec_name, features.heavy_atoms))
    gold_tokens = dict(zip(features.spec_name, features.gold_tokens))
    group = dict(zip(features.spec_name, features.group))
    blocked = set(forbidden) | set(mask.special_token_ids)
    token_masses = constraint.token_masses.numpy()
    open_ids = np.array(
        [index for index in range(len(token_masses)) if index not in blocked]
    )
    hydrogen_fragment = re.compile(r"^\[[0-9]*H[A-Za-z0-9+-]*\]$")
    probes = []
    started = time.perf_counter()
    selected = [
        row
        for row in rows
        if row["sample_dead_ends"]
        and (scope == "all" or group[row["spec_name"]] == "all_dead_end")
    ]
    for index, row in enumerate(selected):
        target = row["neutral_mass"]
        tolerance = mask.ppm_tolerance * 1e-6 * target
        for dead_end in row["sample_dead_ends"]:
            prefix = dead_end["safe"]
            syntax = mask._valid_token_ids(prefix)
            reachable = mask._mass_reachable_token_ids(prefix, target)
            state = _scan(prefix)
            scanned_mass = (
                float(sum(state.atom_masses.values()))
                if state is not None
                else float("nan")
            )
            minimum_mass = (
                state.minimum_mass(valence_slack) if state is not None else float("nan")
            )
            # The mass-shell prune of the run, replayed on the state the run
            # recorded, so a dead end with support left in the grammar can still
            # be attributed.
            shell_allowed = int(
                (
                    dead_end["heavy_mass"] + constraint.token_masses.numpy()
                    <= target + tolerance
                ).sum()
            )
            shell_allowed_ids = {
                token_id
                for token_id in np.nonzero(
                    dead_end["heavy_mass"] + constraint.token_masses.numpy()
                    <= target + tolerance
                )[0].tolist()
                if token_id not in blocked
            }
            # What the mass-shell prune of the run still had room for, so a dead
            # end can be read as "budget spent" versus "budget open".
            residual = target + tolerance - dead_end["heavy_mass"]
            open_masses = token_masses[open_ids]
            fitting = open_masses[(open_masses > 0) & (open_masses <= residual)]
            fragments = [fragment for fragment in prefix.split(".") if fragment]
            trailing_hydrogen = 0
            for fragment in reversed(fragments):
                if hydrogen_fragment.match(fragment):
                    trailing_hydrogen += 1
                else:
                    break
            if not syntax:
                classification = "grammar_deadlock"
            elif not reachable:
                classification = "mass_reachability_deadlock"
            elif not set(reachable) & shell_allowed_ids:
                classification = "token_prune_deadlock"
            else:
                classification = "unexplained"
            probes.append(
                {
                    "spec_name": row["spec_name"],
                    "group": group[row["spec_name"]],
                    "prefix": prefix,
                    "prefix_chars": len(prefix),
                    "neutral_mass": target,
                    "syntax_support": len(syntax),
                    "mass_reachable_support": len(reachable),
                    "shell_allowed_support": shell_allowed,
                    "shell_residual": residual,
                    "shell_fitting_tokens": int(fitting.size),
                    "shell_fitting_min_mass": float(fitting.min())
                    if fitting.size
                    else float("nan"),
                    "shell_fitting_only_hydrogen": bool(
                        fitting.size and float(fitting.max()) < 2.0
                    ),
                    "trailing_hydrogen_fragments": trailing_hydrogen,
                    "classification": classification,
                    "recorded_heavy_mass": dead_end["heavy_mass"],
                    "recorded_heavy_atoms": dead_end["heavy_atoms"],
                    "scanned_heavy_mass": scanned_mass,
                    "scanned_heavy_atoms": len(state.atom_masses) if state else -1,
                    "minimum_mass": minimum_mass,
                    "heavy_mass_fraction": scanned_mass / target,
                    "minimum_mass_fraction": minimum_mass / target,
                    "overshoot": bool(minimum_mass > target + tolerance),
                    "open_ring_labels": len(state.open_rings) if state else -1,
                    "branch_depth": state.branch_depth if state else -1,
                    "expect_atom": bool(state.expect_atom) if state else None,
                    "gold_heavy_atoms": int(gold_atoms[row["spec_name"]]),
                    "gold_tokens": int(gold_tokens[row["spec_name"]]),
                    "atom_fraction": (len(state.atom_masses) if state else 0)
                    / int(gold_atoms[row["spec_name"]]),
                    **(
                        phantom_atoms(state)
                        if state is not None
                        else {
                            "atoms": -1,
                            "zero_mass_atoms": -1,
                            "bracket_hydrogen_atoms": -1,
                            "foreign_element_atoms": -1,
                            "zero_mass_symbols": [],
                        }
                    ),
                }
            )
        if (index + 1) % 10 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"probed {index + 1}/{len(selected)} spectra "
                f"({len(probes)} prefixes, {elapsed:.0f}s)",
                flush=True,
            )
    return probes


HYDROGEN_FRAGMENT = re.compile(r"^\[[0-9]*H[A-Za-z0-9+-]*\]$")
HYDROGEN_MASS = 1.00782503223


def funnel_probe(
    frame: pd.DataFrame, mask, constraint, forbidden: tuple[int, ...], sample: int
) -> dict:
    """Ask what the run could still write just before the hydrogen tail started.

    A dead end whose prefix ends in bracket-hydrogen fragments is cut back to the
    first of them and the support is asked again there, so "the model chose to
    write [H]" can be told apart from "the constraints left nothing else".
    """
    blocked = set(forbidden) | set(mask.special_token_ids)
    token_masses = constraint.token_masses.numpy()
    open_ids = [index for index in range(len(token_masses)) if index not in blocked]
    candidates = frame[frame.trailing_hydrogen_fragments >= 2].head(sample)
    records = []
    for _, probe in candidates.iterrows():
        fragments = [part for part in probe.prefix.split(".") if part]
        keep = len(fragments) - int(probe.trailing_hydrogen_fragments)
        if keep <= 0:
            continue
        truncated = ".".join(fragments[:keep]) + "."
        hydrogen_mass = HYDROGEN_MASS * int(probe.trailing_hydrogen_fragments)
        heavy_mass = probe.recorded_heavy_mass - hydrogen_mass
        residual = probe.neutral_mass * (1 + mask.ppm_tolerance * 1e-6) - heavy_mass
        shell_allowed = {
            token_id for token_id in open_ids if token_masses[token_id] <= residual
        }
        reachable = set(mask._mass_reachable_token_ids(truncated, probe.neutral_mass))
        support = sorted(reachable & shell_allowed)
        hydrogen_support = [
            token_id
            for token_id in support
            if abs(token_masses[token_id] - HYDROGEN_MASS) < 1e-6
        ]
        heavier_support = [
            token_id for token_id in support if token_masses[token_id] > 2.0
        ]
        records.append(
            {
                "spec_name": probe.spec_name,
                "truncated_prefix": truncated,
                "trailing_hydrogen_fragments": int(probe.trailing_hydrogen_fragments),
                "shell_residual": round(float(residual), 4),
                "support": len(support),
                "hydrogen_support": len(hydrogen_support),
                "heavier_than_hydrogen_support": len(heavier_support),
                "zero_mass_support": len(support)
                - len(hydrogen_support)
                - len(heavier_support),
            }
        )
    return {
        "n_probed": len(records),
        "n_with_no_heavier_option": sum(
            1 for record in records if record["heavier_than_hydrogen_support"] == 0
        ),
        "n_with_hydrogen_option": sum(
            1 for record in records if record["hydrogen_support"] > 0
        ),
        "records": records,
    }


def gold_admissibility(
    rows: list[dict], features: pd.DataFrame, mask, valence_slack: float
) -> dict:
    """Would the mask accept the gold answer as a finished string?

    This is the necessary condition for the run to be answerable at all: the gold
    SAFE string has to scan, be terminal, and sit inside the mass window the run
    was given. A gold answer that fails here cannot be reached no matter what the
    model proposes.
    """
    from marlin.grammar import _has_hydrogen_only_exact_mass

    group = dict(zip(features.spec_name, features.group))
    error_ppm = dict(zip(features.spec_name, features.gold_mass_error_ppm))
    outside = {
        name: abs(value) > mask.ppm_tolerance for name, value in error_ppm.items()
    }
    counters: dict[str, Counter] = {}
    refused = []
    for row in rows:
        safe = smiles_to_safe(row["target_smiles"])
        state = _scan(safe)
        tolerance = mask.ppm_tolerance * 1e-6 * row["neutral_mass"]
        admitted = (
            state is not None
            and state.terminal
            and _has_hydrogen_only_exact_mass(
                state, row["neutral_mass"], valence_slack, tolerance
            )
        )
        bucket = counters.setdefault(group[row["spec_name"]], Counter())
        bucket["gold_answers"] += 1
        bucket["admitted"] += int(admitted)
        bucket["outside_mass_window"] += int(outside[row["spec_name"]])
        if not admitted:
            refused.append(
                {
                    "spec_name": row["spec_name"],
                    "group": group[row["spec_name"]],
                    "scans": state is not None,
                    "terminal": bool(state.terminal) if state else False,
                    "gold_mass_error_ppm": round(float(error_ppm[row["spec_name"]]), 2),
                    "safe": safe,
                }
            )
    return {
        "by_group": {name: dict(counter) for name, counter in counters.items()},
        "refused": refused,
    }


def gold_walk(
    rows: list[dict], features: pd.DataFrame, mask, tokenizer, sample: int
) -> dict:
    """Replay the gold token sequence through the mask, token by token.

    The lightest all-dead-end targets are walked because they are the cheapest,
    together with every all-dead-end target carrying a formal charge, which is
    where the run's target mass and the gold molecule disagree.
    """
    table = features.set_index("spec_name")
    dead = [
        row for row in rows if table.loc[row["spec_name"], "group"] == "all_dead_end"
    ]
    lightest = sorted(dead, key=lambda row: row["neutral_mass"])[:sample]
    charged = [row for row in dead if table.loc[row["spec_name"], "formal_charge"] != 0]
    selected = {row["spec_name"]: row for row in lightest + charged}
    records = []
    for row in selected.values():
        safe = smiles_to_safe(row["target_smiles"])
        token_ids = tokenizer.encode(safe, add_special_tokens=False) + [
            tokenizer.eos_token_id
        ]
        refused_at = None
        for index in range(len(token_ids)):
            prefix = tokenizer.decode(token_ids[:index], skip_special_tokens=True)
            allowed = mask._mass_reachable_token_ids(prefix, row["neutral_mass"])
            if token_ids[index] not in allowed:
                refused_at = {
                    "position": index,
                    "token": tokenizer.convert_ids_to_tokens(token_ids[index]),
                    "prefix": prefix,
                    "support": len(allowed),
                }
                break
        records.append(
            {
                "spec_name": row["spec_name"],
                "neutral_mass": row["neutral_mass"],
                "formal_charge": int(table.loc[row["spec_name"], "formal_charge"]),
                "gold_mass_error_ppm": round(
                    float(table.loc[row["spec_name"], "gold_mass_error_ppm"]), 2
                ),
                "gold_tokens": len(token_ids),
                "refused": refused_at,
            }
        )
    return {
        "n_walked": len(records),
        "n_refused": sum(1 for record in records if record["refused"]),
        "n_refused_inside_mass_window": sum(
            1
            for record in records
            if record["refused"]
            and abs(record["gold_mass_error_ppm"]) <= mask.ppm_tolerance
        ),
        "records": records,
    }


def gold_vocabulary_audit(cache_dir: Path, splits: list[str], tokenizer) -> dict:
    """Report which vocabulary entries the gold answers actually need.

    A restriction can only be called safe if every gold answer survives it, so
    the partial bracket spellings are counted against the gold token usage of the
    whole adaptation set rather than against the test split alone.
    """
    token_strings = [
        tokenizer.convert_ids_to_tokens(index) for index in range(len(tokenizer))
    ]
    partial_bracket = [
        index
        for index, token in enumerate(token_strings)
        if (token.startswith("[") and "]" not in token)
        or ("]" in token and not token.startswith("["))
    ]
    used: Counter = Counter()
    encoded = 0
    failures = 0
    for split in splits:
        table = pd.read_csv(cache_dir / split / "metadata.csv")
        for smiles in table["smiles"].astype(str):
            safe = smiles_to_safe(smiles)
            if not safe:
                failures += 1
                continue
            encoded += 1
            used.update(tokenizer.encode(safe, add_special_tokens=False))
    return {
        "splits": splits,
        "gold_answers_encoded": encoded,
        "gold_answers_failed": failures,
        "distinct_tokens_used": len(used),
        "tokens_used": sorted(token_strings[index] for index in used),
        "partial_bracket_tokens_in_vocabulary": len(partial_bracket),
        "partial_bracket_tokens_used_by_gold": sum(
            1 for index in partial_bracket if used[index]
        ),
    }


BRACKET_ATOM = re.compile(r"\[(\d*)([A-Za-z][a-z]?)")


def emission_leak_audit(rows: list[dict]) -> dict:
    """Count the emitted strings that carry a spelling the run declared forbidden.

    ``--forbid-isotope-tokens`` and ``--restrict-organic-elements`` withhold whole
    vocabulary entries, but the vocabulary also holds bracket fragments ("[",
    "[N", "a-]", "g]", digits), so a forbidden atom can still be spelled across
    several allowed tokens. This counts how often that happened.
    """

    def audit(strings: list[str]) -> dict:
        isotopes = 0
        foreign = 0
        elements: set[str] = set()
        for text in strings:
            hits = BRACKET_ATOM.findall(text)
            if any(isotope for isotope, _ in hits):
                isotopes += 1
            outside = {
                symbol
                for _, symbol in hits
                if symbol not in ORGANIC and symbol.capitalize() not in ORGANIC
            }
            if outside:
                foreign += 1
                elements |= outside
        return {
            "n_strings": len(strings),
            "with_isotope_spelling": isotopes,
            "with_non_organic_element": foreign,
            "non_organic_elements": sorted(elements),
        }

    return {
        "dead_end_prefixes": audit(
            [dead["safe"] for row in rows for dead in row["sample_dead_ends"]]
        ),
        "terminal_safes": audit(
            [safe for row in rows for safe in row["sample_terminal_safes"]]
        ),
        "accepted_candidates": audit(
            [candidate["safe"] for row in rows for candidate in row["candidates"]]
        ),
    }


def accepted_candidate_summary(rows: list[dict]) -> dict:
    """Phantom-atom baseline over the SAFE strings the run actually accepted."""
    counts = []
    for row in rows:
        for candidate in row["candidates"]:
            state = _scan(candidate["safe"])
            if state is None:
                counts.append(None)
                continue
            counts.append(phantom_atoms(state))
    scanned = [entry for entry in counts if entry is not None]
    return {
        "n_candidates": len(counts),
        "unscannable": sum(1 for entry in counts if entry is None),
        "with_zero_mass_atom": sum(
            1 for entry in scanned if entry["zero_mass_atoms"] > 0
        ),
        "with_bracket_hydrogen": sum(
            1 for entry in scanned if entry["bracket_hydrogen_atoms"] > 0
        ),
        "with_foreign_element": sum(
            1 for entry in scanned if entry["foreign_element_atoms"] > 0
        ),
    }


def classify_parse_failure(safe: str) -> dict:
    """Bucket why a terminal SAFE string yields no molecule.

    The raw SAFE string is parsed first and its RDKit log captured, because
    ``safe_to_smiles`` silences RDKit internally; only then is the decoder run,
    so a string RDKit accepts but the SAFE decoder rejects is a distinct bucket
    instead of an unexplained one.
    """
    rdBase.LogToPythonStderr()
    buffer = io.StringIO()
    with contextlib.redirect_stderr(buffer):
        parsed = Chem.MolFromSmiles(safe)
    message = " ".join(buffer.getvalue().split())
    smiles = safe_to_smiles(safe, fix=False)
    if smiles and Chem.MolFromSmiles(smiles) is not None:
        return {"bucket": "parses", "message": message[:240]}
    if parsed is not None:
        # RDKit read the string; the SAFE decoder or its standardiser refused it.
        return {"bucket": "safe_decoder", "message": message[:240]}
    bucket = "other"
    for name, pattern in PARSE_BUCKETS:
        if pattern.search(message):
            bucket = name
            break
    return {"bucket": bucket, "message": message[:240]}


def probe_parse_failures(rows: list[dict], features: pd.DataFrame) -> list[dict]:
    group = dict(zip(features.spec_name, features.group))
    probes = []
    for row in rows:
        if group[row["spec_name"]] != "no_parse":
            continue
        for safe in row["sample_terminal_safes"]:
            verdict = classify_parse_failure(safe)
            state = _scan(safe)
            fixed = safe_to_smiles(safe, fix=True)
            probes.append(
                {
                    "spec_name": row["spec_name"],
                    "safe": safe,
                    "bucket": verdict["bucket"],
                    "message": verdict["message"],
                    **(
                        phantom_atoms(state)
                        if state is not None
                        else {
                            "atoms": -1,
                            "zero_mass_atoms": -1,
                            "bracket_hydrogen_atoms": -1,
                            "foreign_element_atoms": -1,
                            "zero_mass_symbols": [],
                        }
                    ),
                    "grammar_scans": state is not None,
                    "grammar_terminal": bool(state.terminal) if state else False,
                    "open_ring_labels": len(state.open_rings) if state else -1,
                    "branch_depth": state.branch_depth if state else -1,
                    "incomplete_token": bool(state.incomplete_token) if state else None,
                    "fixed_parses": bool(
                        fixed and Chem.MolFromSmiles(fixed) is not None
                    ),
                    "max_length_terminated": row["max_length_terminated"],
                    "eos_terminated": row["eos_terminated"],
                }
            )
    return probes


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line) for line in args.predictions.read_text().splitlines() if line
    ]
    tokenizer, mask, constraint, forbidden = build_mask(
        args.cache_dir, args.ppm_tolerance, args.valence_slack
    )
    features = target_features(
        rows,
        args.cache_dir,
        args.split,
        args.fingerprint_key,
        args.threshold,
        tokenizer,
    )
    columns = [
        "neutral_mass",
        "heavy_atoms",
        "rings",
        "aromatic_rings",
        "hetero_fraction",
        "nitrogen_fraction",
        "oxygen_fraction",
        "nitrogens",
        "rotatable_bonds",
        "safe_fragments",
        "gold_tokens",
        "hydrogen_mass_fraction",
        "fingerprint_tanimoto",
        "fingerprint_recall",
        "predicted_bits",
        "true_bits",
    ]
    report = {
        "predictions": str(args.predictions),
        "cache_dir": str(args.cache_dir),
        "group_sizes": features.group.value_counts().to_dict(),
        "group_medians": features.groupby("group")[columns].median().round(4).to_dict(),
        "separation": separation_table(features, columns),
        "separation_dead_vs_mass_miss": separation_table(
            features, columns, "all_dead_end", "mass_miss"
        ),
        "separation_mass_miss_vs_returned": separation_table(
            features, columns, "mass_miss", "returned"
        ),
        "dead_rate_by_quintile": {
            key: features.assign(
                dead_rate=features.dead_ends / 8,
                quintile=pd.qcut(features[key].rank(method="first"), 5, labels=False),
            )
            .groupby("quintile")[["dead_rate", key]]
            .mean()
            .round(4)
            .to_dict()
            for key in ("fingerprint_recall", "neutral_mass", "rings")
        },
        "stratified": [
            stratified_table(features, "fingerprint_recall", "neutral_mass"),
            stratified_table(features, "fingerprint_recall", "rings"),
            stratified_table(features, "nitrogen_fraction", "rings"),
        ],
        "forbidden_token_count": len(forbidden),
        "emission_leak": emission_leak_audit(rows),
        "gold_vocabulary": gold_vocabulary_audit(
            args.cache_dir, args.gold_vocabulary_splits.split(","), tokenizer
        ),
        "gold_admissibility": gold_admissibility(
            rows, features, mask, args.valence_slack
        ),
    }
    if args.gold_walk_sample:
        report["gold_walk"] = gold_walk(
            rows, features, mask, tokenizer, args.gold_walk_sample
        )
    parse_probes = probe_parse_failures(rows, features)
    report["parse_failures"] = {
        "n_spectra": int((features.group == "no_parse").sum()),
        "n_strings": len(parse_probes),
        "buckets": dict(Counter(probe["bucket"] for probe in parse_probes)),
        "grammar_terminal": int(
            sum(probe["grammar_terminal"] for probe in parse_probes)
        ),
        "open_ring_labels": dict(
            Counter(probe["open_ring_labels"] for probe in parse_probes)
        ),
        "fixed_parses": int(sum(probe["fixed_parses"] for probe in parse_probes)),
        "with_zero_mass_atom": int(
            sum(probe["zero_mass_atoms"] > 0 for probe in parse_probes)
        ),
        "with_foreign_element": int(
            sum(probe["foreign_element_atoms"] > 0 for probe in parse_probes)
        ),
        "bucket_by_grammar_terminal": {
            f"{bucket}|terminal={terminal}": count
            for (bucket, terminal), count in Counter(
                (probe["bucket"], probe["grammar_terminal"]) for probe in parse_probes
            ).items()
        },
        "examples": parse_probes[:40],
    }
    report["accepted_candidates"] = accepted_candidate_summary(
        [row for row in rows if row["candidate_returned"]]
    )
    print(json.dumps(report["parse_failures"]["buckets"], indent=1), flush=True)
    if args.reuse_dead_ends is not None:
        frame = pd.read_csv(args.reuse_dead_ends)
        frame["zero_mass_symbols"] = frame.zero_mass_symbols.apply(literal_eval)
    else:
        frame = pd.DataFrame(
            probe_dead_ends(
                rows,
                features,
                mask,
                constraint,
                forbidden,
                args.valence_slack,
                args.dead_end_scope,
            )
        )
    report["dead_ends"] = {
        "scope": args.dead_end_scope,
        "n_prefixes": len(frame),
        "n_spectra": int(frame.spec_name.nunique()),
        "classification": frame.classification.value_counts().to_dict(),
        "syntax_support_zero": int((frame.syntax_support == 0).sum()),
        "mass_support_zero": int((frame.mass_reachable_support == 0).sum()),
        "syntax_support_quantiles": frame.syntax_support.quantile(
            [0, 0.25, 0.5, 0.75, 1]
        ).to_dict(),
        "mass_fraction_quantiles": frame.heavy_mass_fraction.quantile(
            [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
        )
        .round(4)
        .to_dict(),
        "minimum_mass_fraction_quantiles": frame.minimum_mass_fraction.quantile(
            [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
        )
        .round(4)
        .to_dict(),
        "overshoot": int(frame.overshoot.sum()),
        "atom_fraction_quantiles": frame.atom_fraction.quantile(
            [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
        )
        .round(4)
        .to_dict(),
        "open_ring_label_histogram": frame.open_ring_labels.value_counts()
        .sort_index()
        .to_dict(),
        "mass_atom_disagreement": int(
            (frame.recorded_heavy_atoms != frame.scanned_heavy_atoms).sum()
        ),
        "mass_value_disagreement": int(
            (abs(frame.recorded_heavy_mass - frame.scanned_heavy_mass) > 0.01).sum()
        ),
        "with_zero_mass_atom": int((frame.zero_mass_atoms > 0).sum()),
        "with_bracket_hydrogen": int((frame.bracket_hydrogen_atoms > 0).sum()),
        "with_foreign_element": int((frame.foreign_element_atoms > 0).sum()),
        "zero_mass_atom_histogram": frame.zero_mass_atoms.value_counts()
        .sort_index()
        .to_dict(),
        "zero_mass_symbol_counts": dict(
            Counter(symbol for symbols in frame.zero_mass_symbols for symbol in symbols)
        ),
        "shell_budget": {
            "no_positive_mass_token_fits": int((frame.shell_fitting_tokens == 0).sum()),
            "only_hydrogen_mass_fits": int(frame.shell_fitting_only_hydrogen.sum()),
            "shell_residual_quantiles": frame.shell_residual.quantile(
                [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
            )
            .round(4)
            .to_dict(),
            "trailing_hydrogen_fragment_histogram": frame.trailing_hydrogen_fragments.value_counts()
            .sort_index()
            .to_dict(),
        },
        "mass_fraction_by_phantom": frame.groupby(frame.zero_mass_atoms > 0)[
            ["heavy_mass_fraction", "minimum_mass_fraction", "atom_fraction"]
        ]
        .median()
        .round(4)
        .to_dict(),
    }
    if args.funnel_sample:
        report["funnel"] = funnel_probe(
            frame, mask, constraint, forbidden, args.funnel_sample
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output.with_suffix(".dead_ends.csv"), index=False)
    features.to_csv(args.output.with_suffix(".features.csv"), index=False)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report["dead_ends"], indent=1, default=str))


if __name__ == "__main__":
    main()
