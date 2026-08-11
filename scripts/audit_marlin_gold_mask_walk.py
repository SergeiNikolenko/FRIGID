#!/usr/bin/env python3
"""Walk every gold answer of a split token by token through the decoding mask.

`scripts/audit_marlin_terminal_hydrogens.py` asks whether the *finished* gold
string is accepted at EOS. This asks the stricter question the decoder actually
faces: at every position of the gold tokenisation, is the gold token inside the
mask's support? One rejected position means the decoder can never write that
molecule, whatever the model predicts, so the count of rejected positions is the
acceptance gate for any change to the grammar, the mass shell or the token
property table.

The mask is built exactly as `scripts/evaluate_marlin_nplib1.py` builds it,
including the chemistry restrictions, and the walk goes through
:meth:`marlin.grammar.SafeGrammarMask.admits`, which shares its code path with
the sampler's ``__call__``.

Two conditioning-mass conventions are available, because a permanently charged
target distinguishes them:

* ``--target-mass molecule`` uses RDKit's exact mass of the gold molecule;
* ``--target-mass conditioning`` (the default) uses the neutral-equivalent mass
  the pipeline conditions on, which subtracts a proton per unit of formal charge
  so that a cation written as ``[N+]`` explains the same precursor m/z as the
  neutral species the run assumed.

Usage:
  PYTHONPATH=src python scripts/audit_marlin_gold_mask_walk.py \
      --metadata .../test/metadata.csv --metadata .../train/metadata.csv \
      --tokenizer .../tokenizer.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from dlm.utils.utils_chem import smiles_to_safe
from marlin.grammar import SafeGrammarMask
from marlin.mass_shell import PROTON_MASS, conditioning_mass
from marlin.tokenizer import load_safe_tokenizer
from marlin.token_properties import foreign_element_token_ids, isotope_token_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, action="append", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--mass-column", default="neutral_mass")
    parser.add_argument("--ppm-tolerance", type=float, default=10.0)
    parser.add_argument("--valence-slack", type=float, default=4.0)
    parser.add_argument(
        "--target-mass",
        choices=("conditioning", "molecule"),
        default="conditioning",
    )
    parser.add_argument("--no-chemistry-restrictions", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_mask(
    tokenizer,
    ppm_tolerance: float,
    valence_slack: float,
    chemistry_restrictions: bool,
) -> SafeGrammarMask:
    token_strings = [
        tokenizer.convert_ids_to_tokens(index) for index in range(len(tokenizer))
    ]
    special_ids = (
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.mask_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
    )
    chemistry_forbidden_ids = (
        tuple(
            sorted(
                set(isotope_token_ids(token_strings))
                | set(foreign_element_token_ids(token_strings))
            )
        )
        if chemistry_restrictions
        else ()
    )
    return SafeGrammarMask(
        token_strings,
        lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
        eos_token_id=tokenizer.eos_token_id,
        mask_token_id=tokenizer.mask_token_id,
        special_token_ids=special_ids,
        forbidden_token_ids=chemistry_forbidden_ids,
        ppm_tolerance=ppm_tolerance,
        valence_slack=valence_slack,
        mass_reachability_prune=True,
        # Both halves of the restriction, so the walk holds the gold answers to
        # the element and isotope rules the state enforces as well as to the
        # token block list the caller passes.
        restrict_organic_elements=chemistry_restrictions,
        forbid_isotopes=chemistry_restrictions,
    )


def walk_gold(
    mask: SafeGrammarMask,
    token_ids: list[int],
    target_mass: float,
) -> int | None:
    """Return the first position the mask refuses, or ``None`` if it admits all."""
    prefix: list[int] = []
    for position, token_id in enumerate(token_ids):
        if not mask.admits(prefix, token_id, target_mass):
            return position
        prefix.append(token_id)
    if not mask.admits(prefix, mask.eos_token_id, target_mass):
        return len(token_ids)
    return None


def audit_split(
    metadata: Path,
    tokenizer,
    mask: SafeGrammarMask,
    args: argparse.Namespace,
) -> dict:
    table = pd.read_csv(metadata)
    if args.limit:
        table = table.iloc[: args.limit]
    special_ids = {
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.mask_token_id,
    }
    positions = 0
    targets = 0
    rejections: list[dict] = []
    scan_failures: list[dict] = []
    charged: list[dict] = []
    inside_window_molecule = 0
    inside_window_conditioning = 0
    window_rows = 0

    for row_index, row in enumerate(table.itertuples(index=False)):
        smiles = str(getattr(row, args.smiles_column))
        molecule = Chem.MolFromSmiles(smiles)
        safe = smiles_to_safe(smiles)
        if molecule is None or safe is None:
            scan_failures.append({"row": row_index, "smiles": smiles, "safe": safe})
            continue
        charge = Chem.GetFormalCharge(molecule)
        exact_mass = ExactMolWt(molecule)
        target_mass = (
            exact_mass
            if args.target_mass == "molecule"
            else conditioning_mass(molecule)
        )

        # How the gold answer compares with the conditioning mass the run
        # recorded for its spectrum, under both conventions.
        run_mass = getattr(row, args.mass_column, None)
        if run_mass is not None and float(run_mass) > 0:
            run_mass = float(run_mass)
            window = args.ppm_tolerance * 1e-6 * run_mass
            window_rows += 1
            inside_window_molecule += abs(exact_mass - run_mass) <= window
            inside_window_conditioning += (
                abs(conditioning_mass(molecule) - run_mass) <= window
            )

        token_ids = [
            token_id
            for token_id in tokenizer(safe)["input_ids"]
            if token_id not in special_ids
        ]
        targets += 1
        positions += len(token_ids) + 1
        refused_at = walk_gold(mask, token_ids, target_mass)
        record = {
            "row": row_index,
            "smiles": smiles,
            "safe": safe,
            "formal_charge": charge,
            "target_mass": target_mass,
            "tokens": len(token_ids),
            "refused_at": refused_at,
        }
        if charge:
            charged.append(record)
        if refused_at is not None:
            rejections.append(record)

    return {
        "metadata": str(metadata),
        "targets": targets,
        "positions": positions,
        "scan_failures": len(scan_failures),
        "gold_rejections": len(rejections),
        "rejection_examples": rejections[:10],
        "charged_targets": len(charged),
        "charged_rejections": sum(
            1 for record in charged if record["refused_at"] is not None
        ),
        "conditioning_mass_window": {
            "rows": window_rows,
            "inside_window_molecule_mass": inside_window_molecule,
            "inside_window_conditioning_mass": inside_window_conditioning,
        },
    }


def main() -> None:
    args = parse_args()
    tokenizer = load_safe_tokenizer(args.tokenizer)
    mask = build_mask(
        tokenizer,
        args.ppm_tolerance,
        args.valence_slack,
        not args.no_chemistry_restrictions,
    )
    splits = [audit_split(metadata, tokenizer, mask, args) for metadata in args.metadata]
    manifest = {
        "kind": "MARLIN gold-answer walk through the decoding mask",
        "git_commit": _git_commit(),
        "tokenizer": str(args.tokenizer),
        "ppm_tolerance": args.ppm_tolerance,
        "valence_slack": args.valence_slack,
        "target_mass": args.target_mass,
        "proton_mass": PROTON_MASS,
        "chemistry_restrictions": not args.no_chemistry_restrictions,
        "splits": splits,
        "totals": {
            "targets": sum(split["targets"] for split in splits),
            "positions": sum(split["positions"] for split in splits),
            "gold_rejections": sum(split["gold_rejections"] for split in splits),
            "scan_failures": sum(split["scan_failures"] for split in splits),
            "charged_targets": sum(split["charged_targets"] for split in splits),
            "charged_rejections": sum(split["charged_rejections"] for split in splits),
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
