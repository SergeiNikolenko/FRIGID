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


def draw_environment(molecule: Chem.Mol, atom: int, radius: int) -> str:
    """Draw the whole molecule with one Morgan environment lit up."""
    bonds = (
        list(Chem.FindAtomEnvironmentOfRadiusN(molecule, radius, atom))
        if radius
        else []
    )
    atoms = {atom}
    for bond_index in bonds:
        bond = molecule.GetBondWithIdx(bond_index)
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())

    rdDepictor.Compute2DCoords(molecule)
    drawer = rdMolDraw2D.MolDraw2DSVG(150, 110)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.bondLineWidth = 1
    options.setAtomPalette(PALETTE)
    for element, colour in ELEMENTS:
        options.updateAtomPalette({element: colour})
    options.setHighlightColour((0.10, 0.62, 0.44, 0.55))
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, molecule, highlightAtoms=sorted(atoms), highlightBonds=bonds
    )
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
            if bit in info:
                atom, radius = info[bit][0]
                svg = draw_environment(molecule, int(atom), int(radius))
                drawn += 1
            entries.append(
                {
                    "bit": int(bit),
                    "kind": kind,
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
