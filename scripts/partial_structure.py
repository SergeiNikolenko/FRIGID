"""Draw what the decoder has written, however unfinished it is.

A prefix of a SAFE string is almost never a molecule: it sits inside an open
ring, an open branch, or a half-written bracket atom. Refusing to draw those
positions leaves the reader staring at an empty frame for most of a walk, so
this falls back in three steps and always says which one it took:

* ``valid``   -- the prefix, or its longest readable head, parses as written;
* ``partial`` -- it parses once unclosed rings and branches are patched and
                 sanitisation is relaxed, so valences and aromaticity may be
                 impossible; the drawing is still the atoms and bonds written;
* ``atoms``   -- nothing parses at all, so the atoms are drawn on their own,
                 unbonded, in the order they were written.

Only the third case loses information, and it loses only the bonds.
"""

from __future__ import annotations

import re

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdDepictor
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
ATOM = re.compile(r"\[[^\]]*\]|Br|Cl|Si|Se|[BCNOPSFIbcnops]")
RING = re.compile(r"%\d{2}|\d")


def repair(text: str) -> str:
    """Close what the decoder left open, so a prefix can be parsed at all."""
    text = text.rstrip("=#-/\\.(")
    # an unterminated bracket atom cannot be repaired, only dropped
    if text.count("[") > text.count("]"):
        text = text[: text.rindex("[")]
    # ring bonds opened once and never closed
    outside = re.sub(r"\[[^\]]*\]", "", text)
    for label in {label for label in RING.findall(outside) if outside.count(label) % 2}:
        text = text.replace(label, "", 1)
    text += ")" * (text.count("(") - text.count(")"))
    # a branch that was opened and never filled leaves "()", which parses nowhere
    while "()" in text:
        text = text.replace("()", "")
    return text.rstrip("=#-/\\.(")


def atoms_only(text: str) -> Chem.Mol | None:
    """The written atoms with no bonds at all -- the last thing worth drawing."""
    builder = Chem.RWMol()
    for symbol in ATOM.findall(text):
        try:
            piece = Chem.MolFromSmiles(symbol if symbol.startswith("[") else f"[{symbol}]")
        except Exception:
            piece = None
        if piece is None or piece.GetNumAtoms() != 1:
            continue
        atom = Chem.Atom(piece.GetAtomWithIdx(0).GetAtomicNum())
        atom.SetNoImplicit(True)
        builder.AddAtom(atom)
    return builder.GetMol() if builder.GetNumAtoms() else None


def best_effort(written: str, to_smiles) -> tuple[Chem.Mol | None, str, str]:
    """Return the most complete molecule this prefix supports, and how it was got."""
    with rdBase.BlockLogs():
        for cut in range(len(written), 0, -1):
            piece = written[:cut].rstrip(".")
            if not piece:
                continue
            smiles = to_smiles(piece)
            candidate = Chem.MolFromSmiles(smiles) if smiles else None
            if candidate is None:
                candidate = Chem.MolFromSmiles(piece)
                smiles = piece if candidate is not None else smiles
            if candidate is not None:
                if cut == len(written):
                    return candidate, "valid", smiles or piece
                break

        patched = repair(written)
        for text in (patched, to_smiles(patched) or ""):
            if not text:
                continue
            molecule = Chem.MolFromSmiles(text, sanitize=False)
            if molecule is None:
                continue
            molecule.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(
                molecule,
                sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
                ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
                catchErrors=True,
            )
            if molecule.GetNumAtoms():
                # Hydrogens the decoder has not written are not the decoder's:
                # without this a lone aromatic carbon is drawn as methane.
                for atom in molecule.GetAtoms():
                    atom.SetNoImplicit(True)
                    atom.SetNumExplicitHs(0)
                return molecule, "partial", text

        loose = atoms_only(written)
        if loose is not None:
            return loose, "atoms", written
    return None, "none", ""


def draw(molecule: Chem.Mol, width: int = 240, height: int = 175) -> str | None:
    """Draw a molecule that may not survive sanitisation."""
    if molecule is None or not molecule.GetNumAtoms():
        return None
    try:
        rdDepictor.Compute2DCoords(molecule)
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
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule, kekulize=False)
    except Exception:
        try:
            drawer.DrawMolecule(molecule)
        except Exception:
            return None
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace(
        "<?xml version='1.0' encoding='iso-8859-1'?>", ""
    )
