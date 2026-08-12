#!/usr/bin/env python3
"""Make the conditioning fingerprint readable, bit by bit.

The decoding page draws 4,096 Morgan bits as a grid of dots, which shows how
sparse the conditioning is and nothing else. This script gives each bit that is
on the substructure it stands for, and puts the Morgan fingerprint of the gold
answer next to the fingerprint DreaMS actually predicted from the spectrum, so
the reader can see the input the decoder is really handed:

* ``true``      -- on in the gold Morgan fingerprint and predicted from the spectrum;
* ``missed``    -- on in gold, but the spectrum model did not predict it;
* ``spurious``  -- predicted from the spectrum, absent from gold.

Usage:
  PYTHONPATH=src python scripts/enrich_bit_panel.py docs/decoding-demo/index.html \\
      --dreams .../val/dreams_predictions.npz --metadata .../val/metadata.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor, rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

PALETTE = {-1: (0.83, 0.85, 0.90)}
ELEMENTS = (
    (7, (0.45, 0.68, 1.00)),
    (8, (1.00, 0.47, 0.42)),
    (9, (0.42, 0.85, 0.66)),
    (16, (0.98, 0.78, 0.35)),
    (17, (0.42, 0.85, 0.66)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("page", type=Path)
    parser.add_argument("--dreams", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--max-bits", type=int, default=48)
    return parser.parse_args()


def fragment_of(molecule: Chem.Mol, atom: int, radius: int) -> tuple[Chem.Mol | None, str]:
    """Cut the Morgan environment out as a molecule of its own.

    Drawing the whole molecule with a few atoms highlighted makes every bit look
    the same. The environment itself -- an atom and everything within ``radius``
    bonds of it -- is what the bit actually stands for, so that is what gets
    drawn.
    """
    if radius == 0:
        single = Chem.RWMol()
        source = molecule.GetAtomWithIdx(atom)
        copy = Chem.Atom(source.GetSymbol())
        copy.SetIsAromatic(source.GetIsAromatic())
        copy.SetFormalCharge(source.GetFormalCharge())
        # A radius-0 bit is the atom alone. Without this RDKit fills in the
        # hydrogens and the card reads "CH4" where the bit means "a carbon".
        copy.SetNoImplicit(True)
        single.AddAtom(copy)
        piece = single.GetMol()
        return piece, source.GetSymbol()
    bonds = list(Chem.FindAtomEnvironmentOfRadiusN(molecule, radius, atom))
    if not bonds:
        return None, ""
    piece = Chem.PathToSubmol(molecule, bonds)
    try:
        label = Chem.MolToSmiles(piece)
    except Exception:
        label = ""
    return piece, label


def draw(piece: Chem.Mol, width: int = 150, height: int = 110) -> str | None:
    if piece is None:
        return None
    try:
        rdDepictor.Compute2DCoords(piece)
    except Exception:
        return None
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.bondLineWidth = 2
    options.setAtomPalette(PALETTE)
    for element, colour in ELEMENTS:
        options.updateAtomPalette({element: colour})
    try:
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, piece)
    except Exception:
        drawer.DrawMolecule(piece)
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace(
        "<?xml version='1.0' encoding='iso-8859-1'?>", ""
    )


def main() -> int:
    args = parse_args()
    dreams = np.load(args.dreams)
    probabilities = dreams["probs"]
    spectrum_ids = [str(name) for name in dreams["spectrum_ids"]]
    row_of = {name: index for index, name in enumerate(spectrum_ids)}
    metadata = pd.read_csv(args.metadata)
    index_of = dict(
        zip(metadata["spec_name"].astype(str), metadata["fingerprint_index"])
    )

    lines = args.page.read_text().split("\n")
    line_index = next(
        i for i, line in enumerate(lines) if line.startswith("const DEMO = {")
    )
    data = json.loads(re.match(r"const DEMO = (.*);$", lines[line_index]).group(1))

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    drawn = 0
    for record in data["molecules"]:
        molecule = Chem.MolFromSmiles(record["smiles"])
        if molecule is None:
            continue
        output = rdFingerprintGenerator.AdditionalOutput()
        output.AllocateBitInfoMap()
        fingerprint = generator.GetFingerprint(molecule, additionalOutput=output)
        info = output.GetBitInfoMap()
        gold_bits = set(fingerprint.GetOnBits())

        spec_name = record["spec_name"]
        row = row_of.get(spec_name)
        if row is None and spec_name in index_of:
            row = int(index_of[spec_name])
        predicted = probabilities[row] if row is not None else None
        predicted_bits = (
            set(np.nonzero(predicted >= args.threshold)[0].tolist())
            if predicted is not None
            else set()
        )

        entries = []
        for bit in sorted(gold_bits | predicted_bits):
            probability = float(predicted[bit]) if predicted is not None else None
            if bit in gold_bits and bit in predicted_bits:
                kind = "true"
            elif bit in gold_bits:
                kind = "missed"
            else:
                kind = "spurious"
            svg = None
            label = ""
            radius = None
            if bit in info:
                atom, environment_radius = info[bit][0]
                radius = int(environment_radius)
                piece, label = fragment_of(molecule, int(atom), radius)
                svg = draw(piece)
                if svg is not None:
                    drawn += 1
            entries.append(
                {
                    "bit": int(bit),
                    "kind": kind,
                    "radius": radius,
                    "label": label,
                    "probability": None if probability is None else round(probability, 4),
                    "svg": svg,
                }
            )
        entries.sort(
            key=lambda entry: (
                {"true": 0, "missed": 1, "spurious": 2}[entry["kind"]],
                -(entry["probability"] or 0),
            )
        )

        record["bit_panel"] = {
            "threshold": args.threshold,
            "gold_on": len(gold_bits),
            "predicted_on": len(predicted_bits),
            "true": len(gold_bits & predicted_bits),
            "missed": len(gold_bits - predicted_bits),
            "spurious": len(predicted_bits - gold_bits),
            "has_prediction": predicted is not None,
            "entries": entries[: args.max_bits],
            "shown": min(len(entries), args.max_bits),
            "total": len(entries),
        }

    lines[line_index] = "const DEMO = " + json.dumps(data, separators=(",", ":")) + ";"
    args.page.write_text("\n".join(lines))
    print(f"drew {drawn} bit environments into {args.page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
