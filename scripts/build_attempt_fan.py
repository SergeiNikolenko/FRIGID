#!/usr/bin/env python3
"""Turn a recorded decode into the fan of attempts a spectrum actually took.

``scripts/evaluate_marlin_nplib1.py --trace-output`` writes, for every spectrum,
every token the block decoder committed and how each attempt ended. This script
reads that file and lays the attempts out as one tree: the rows share a prefix
for as long as they agree, split where they stop agreeing, and end in a leaf
that is either a returned candidate or a dead end.

Each dead end is put to the same masks the run used, so it can be named:

* ``grammar_deadlock``          -- nothing may follow this prefix at all;
* ``mass_reachability_deadlock``-- syntax allows tokens, the mass shell does not;
* ``token_prune_deadlock``      -- both allow tokens, but not the same ones.

Usage:
  PYTHONPATH=src python scripts/build_attempt_fan.py \
      --trace /mnt/.../attempt-fan/trace.jsonl \
      --predictions /mnt/.../attempt-fan/predictions.jsonl \
      --tokenizer .../tokenizer.json \
      --html-template docs/attempt-fan/template.html \
      --html-output docs/attempt-fan/index.html \
      --output docs/attempt-fan/attempt_fan.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from dlm.utils.utils_chem import safe_to_smiles
from marlin.grammar import SafeGrammarMask
from marlin.mass_shell import conditioning_mass
from marlin.token_properties import foreign_element_token_ids, isotope_token_ids
from marlin.tokenizer import load_safe_tokenizer

RDLogger.DisableLog("rdApp.*")

ENDINGS = ("accepted", "dead_end", "eos_rejected", "max_length")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--ppm-tolerance", type=float, default=10.0)
    parser.add_argument("--valence-slack", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--html-template", type=Path)
    parser.add_argument("--html-output", type=Path)
    return parser.parse_args()


def draw(smiles: str | None, width: int = 300, height: int = 220) -> str | None:
    if not smiles:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    rdDepictor.Compute2DCoords(molecule)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.bondLineWidth = 2
    # The same palette as the rest of the page: light bonds for a dark surface,
    # which the stylesheet inverts on the light theme.
    options.setAtomPalette({-1: (0.83, 0.85, 0.90)})
    for element, colour in (
        (7, (0.45, 0.68, 1.00)),
        (8, (1.00, 0.47, 0.42)),
        (9, (0.42, 0.85, 0.66)),
        (16, (0.98, 0.78, 0.35)),
        (17, (0.42, 0.85, 0.66)),
    ):
        options.updateAtomPalette({element: colour})
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace(
        "<?xml version='1.0' encoding='iso-8859-1'?>", ""
    )


def readable_prefix(written: str) -> str | None:
    """Largest piece of a written SAFE string RDKit can still read."""
    for cut in range(len(written), 0, -1):
        piece = written[:cut].rstrip(".")
        if not piece:
            continue
        with rdBase.BlockLogs():
            smiles = safe_to_smiles(piece)
            if smiles and Chem.MolFromSmiles(smiles) is not None:
                return smiles
            if Chem.MolFromSmiles(piece) is not None:
                return piece
    return None


def matched_atoms(target: Chem.Mol | None, smiles: str | None) -> list[int]:
    if target is None or not smiles:
        return []
    covered: set[int] = set()
    for piece in sorted(
        (piece for piece in smiles.split(".") if piece), key=len, reverse=True
    ):
        query = Chem.MolFromSmiles(piece)
        if query is None or query.GetNumAtoms() == 0:
            continue
        for match in target.GetSubstructMatches(query, uniquify=False):
            if not covered.intersection(match):
                covered.update(match)
                break
    return sorted(covered)


def classify_dead_end(
    prefix: str, target_mass: float, full_mask: SafeGrammarMask, syntax_mask: SafeGrammarMask
) -> dict[str, object]:
    """Name a dead end with the same masks the run used."""
    syntax = syntax_mask._valid_token_ids(prefix)
    reachable = full_mask._mass_reachable_token_ids(prefix, target_mass)
    if not syntax:
        cause = "grammar_deadlock"
    elif not reachable:
        cause = "mass_reachability_deadlock"
    else:
        cause = "token_prune_deadlock"
    return {
        "cause": cause,
        "syntax_support": len(syntax),
        "mass_reachable_support": len(reachable),
    }


def morgan(molecule: Chem.Mol):
    return AllChem.GetMorganGenerator(radius=2, fpSize=4096).GetFingerprint(molecule)


def main() -> int:
    args = parse_args()
    tokenizer = load_safe_tokenizer(args.tokenizer)
    token_strings = [
        tokenizer.convert_ids_to_tokens(index) for index in range(len(tokenizer))
    ]
    chemistry_forbidden = tuple(
        sorted(
            set(isotope_token_ids(token_strings))
            | set(foreign_element_token_ids(token_strings))
        )
    )
    special_ids = tuple(
        token_id
        for token_id in (
            tokenizer.unk_token_id,
            tokenizer.bos_token_id,
            tokenizer.pad_token_id,
            tokenizer.mask_token_id,
        )
        if token_id is not None
    )

    def build_mask(*, chemistry: bool, mass: bool) -> SafeGrammarMask:
        return SafeGrammarMask(
            token_strings,
            lambda ids: tokenizer.decode(ids, skip_special_tokens=True),
            eos_token_id=tokenizer.eos_token_id,
            mask_token_id=tokenizer.mask_token_id,
            special_token_ids=special_ids,
            forbidden_token_ids=chemistry_forbidden if chemistry else (),
            ppm_tolerance=args.ppm_tolerance,
            valence_slack=args.valence_slack,
            mass_reachability_prune=mass,
            restrict_organic_elements=chemistry,
            forbid_isotopes=chemistry,
        )

    full_mask = build_mask(chemistry=True, mass=True)
    syntax_mask = build_mask(chemistry=False, mass=False)

    predictions = {}
    if args.predictions is not None and args.predictions.exists():
        for line in args.predictions.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                predictions[str(row["spec_name"])] = row

    spectra = []
    for line in args.trace.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        spec_name = str(record["spec_name"])
        target = Chem.MolFromSmiles(record["target_smiles"])
        target_mass = float(record["neutral_mass"])
        prediction = predictions.get(spec_name, {})

        rows: dict[int, list[dict]] = defaultdict(list)
        for entry in record["steps"]:
            rows[int(entry["row"])].append(entry)

        attempts = []
        for row_index in sorted(rows):
            entries = rows[row_index]
            steps = []
            written = ""
            ending = None
            for entry in entries:
                if "ending" in entry:
                    ending = entry
                    continue
                token = token_strings[int(entry["token"])]
                written += token
                steps.append(
                    {
                        "position": int(entry["position"]),
                        "block": int(entry["block"]),
                        "token": token,
                        "support": int(entry["support"]),
                        "confidence": round(float(entry["confidence"]), 5),
                        "heavy_mass": round(float(entry["heavy_mass"]), 4),
                        "alternatives": [
                            [token_strings[int(token_id)], round(float(probability), 5)]
                            for token_id, probability in entry.get("alternatives", [])
                        ],
                    }
                )
            attempt = {
                "row": row_index,
                "steps": steps,
                "written": written,
                "ending": ending.get("ending") if ending else "unfinished",
                "safe": ending.get("safe") if ending else written,
            }
            smiles = ending.get("smiles") if ending else None
            if not smiles:
                smiles = readable_prefix(attempt["safe"] or written)
            attempt["smiles"] = smiles
            attempt["svg"] = draw(smiles)
            lit = matched_atoms(target, smiles)
            attempt["gold_atoms"] = lit
            attempt["gold_share"] = (
                round(len(lit) / target.GetNumHeavyAtoms(), 3) if target else 0.0
            )
            molecule = Chem.MolFromSmiles(smiles) if smiles else None
            if molecule is not None:
                attempt["fragments"] = len(Chem.GetMolFrags(molecule))
                attempt["mass"] = round(float(conditioning_mass(molecule)), 4)
                attempt["mass_error_ppm"] = round(
                    (attempt["mass"] - target_mass) / target_mass * 1e6, 1
                )
                if target is not None:
                    attempt["tanimoto"] = round(
                        DataStructs.TanimotoSimilarity(morgan(target), morgan(molecule)),
                        3,
                    )
            if attempt["ending"] == "dead_end" and ending is not None:
                attempt.update(
                    classify_dead_end(
                        ending.get("safe", ""), target_mass, full_mask, syntax_mask
                    )
                )
                attempt["died_at"] = len(steps)
            attempts.append(attempt)

        spectra.append(
            {
                "spec_name": spec_name,
                "smiles": record["target_smiles"],
                "target_mass": target_mass,
                "formula": Chem.rdMolDescriptors.CalcMolFormula(target)
                if target
                else "",
                "heavy_atoms": target.GetNumHeavyAtoms() if target else 0,
                "target_svg": draw(record["target_smiles"], 340, 250),
                "runtime_seconds": float(record.get("runtime_seconds", 0.0)),
                "candidates_requested": int(record.get("candidates", len(attempts))),
                "returned": len(prediction.get("candidates", [])),
                "attempts": attempts,
            }
        )

    payload = {
        "ppm_tolerance": args.ppm_tolerance,
        "spectra": spectra,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {len(spectra)} spectra to {args.output}")

    if args.html_template is not None and args.html_output is not None:
        template = args.html_template.read_text()
        args.html_output.parent.mkdir(parents=True, exist_ok=True)
        args.html_output.write_text(
            template.replace("/*FAN_DATA*/", json.dumps(payload, separators=(",", ":")))
        )
        print(f"wrote {args.html_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
