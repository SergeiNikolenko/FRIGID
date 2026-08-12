"""Give the decoding page two things RDKit has to compute.

1. Every position gets a drawing, not only the positions where the written
   prefix happens to parse. A prefix mid-token is normalised to the longest
   piece of itself that RDKit can read, so the reader watches the molecule
   grow instead of watching an empty frame.
2. Every drawable prefix gets the gold atoms it covers, so the gold answer can
   light up the part that is already written -- and stay dark when the decoder
   is building something the gold answer does not contain.

Both are derived from the SMILES the record already holds; nothing here needs
the checkpoint or a GPU.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

sys.path.insert(0, "src")
from dlm.utils.utils_chem import safe_to_smiles  # noqa: E402
from partial_structure import best_effort, draw as draw_partial  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def draw(smiles: str, width: int = 260, height: int = 190) -> str | None:
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
    return drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")


def readable_prefix(written: str) -> str | None:
    """Longest prefix of a written SAFE string that RDKit can still read.

    The decoder writes one token at a time, so most positions sit inside an
    unfinished ring or branch. Trimming from the right finds the largest piece
    that is a molecule, which is what the reader should see at that moment.
    """
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


def matched_atoms(target: Chem.Mol, smiles: str) -> list[int]:
    """Gold atom indices covered by the fragments of ``smiles``.

    Fragments are placed largest first and never reuse an atom, so two copies
    of one piece light up two parts of the molecule instead of the same part
    twice. A fragment the gold answer does not contain simply matches nothing.
    """
    covered: set[int] = set()
    fragments = sorted((piece for piece in smiles.split(".") if piece), key=len, reverse=True)
    for piece in fragments:
        query = Chem.MolFromSmiles(piece)
        if query is None or query.GetNumAtoms() == 0:
            continue
        for match in target.GetSubstructMatches(query, uniquify=False):
            if not covered.intersection(match):
                covered.update(match)
                break
    return sorted(covered)


def main() -> int:
    page = Path(sys.argv[1])
    lines = page.read_text().split("\n")
    index = next(i for i, line in enumerate(lines) if line.startswith("const DEMO = {"))
    data = json.loads(re.match(r"const DEMO = (.*);$", lines[index]).group(1))

    drawings = 0
    for molecule in data["molecules"]:
        target = Chem.MolFromSmiles(molecule["smiles"])
        partial_svgs: dict[str, str] = {}
        matches: dict[str, list[int]] = {}
        kinds: dict[str, str] = {}
        labels: dict[str, str] = {}
        for step in molecule["steps"]:
            written = step["written"]
            step["partial"] = written
            if written in partial_svgs:
                continue
            piece, kind, label = best_effort(written, safe_to_smiles)
            drawing = draw_partial(piece, 260, 190)
            if drawing is None:
                step["partial"] = None
                continue
            partial_svgs[written] = drawing
            kinds[written] = kind
            labels[written] = label
            matches[written] = (
                matched_atoms(target, label) if target is not None and kind == "valid" else []
            )
            drawings += 1
        molecule["partial_svgs"] = partial_svgs
        molecule["partial_kinds"] = kinds
        molecule["partial_labels"] = labels
        molecule["gold_matches"] = matches

    lines[index] = "const DEMO = " + json.dumps(data, separators=(",", ":")) + ";"
    page.write_text("\n".join(lines))
    print(f"drew {drawings} intermediate structures into {page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
