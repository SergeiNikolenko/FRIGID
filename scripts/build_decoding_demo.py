#!/usr/bin/env python3
"""Record a teacher-forced decode of gold answers for the decoding demo page.

At every position of a gold SAFE string this records two things side by side:
what the decoder wanted to write, and what the mask allowed it to write. The
model's ranked tokens come from the same ``sampling_logits`` call the sampler
makes, and each one is then put to the mask that the evaluation run uses, so a
refusal in this record is a refusal the decoder actually meets.

A refused token is attributed to the first restriction that rejects it:

* ``chemistry``    -- withheld by token id (isotope or non-organic element);
* ``connectivity`` -- the SAFE separator would strand a finished fragment;
* ``syntax``       -- outside the lexical SAFE support of the prefix;
* ``mass``         -- syntactically fine, but the target mass is then out of reach.

Usage:
  PYTHONPATH=src python scripts/build_decoding_demo.py \
      --checkpoint .../step=100000.ckpt --tokenizer .../tokenizer.json \
      --metadata .../val/metadata.csv --spec-manifest configs/.../panel.tsv \
      --predictions /mnt/.../clean-before/predictions.jsonl \
      --molecules 8 --output docs/decoding-demo/decoding_demo.json
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dlm.utils.utils_chem import safe_to_smiles, smiles_to_safe
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from evaluate_marlin_nplib1 import load_decoder
from marlin.grammar import SafeGrammarMask, _scan
from marlin.mass_shell import conditioning_mass
from marlin.token_properties import foreign_element_token_ids, isotope_token_ids
from marlin.tokenizer import load_safe_tokenizer

RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--spec-manifest", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--molecules", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--ppm-tolerance", type=float, default=10.0)
    parser.add_argument("--valence-slack", type=float, default=4.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--html-template", type=Path)
    parser.add_argument("--html-output", type=Path)
    return parser.parse_args()


def draw(smiles: str | None, width: int = 340, height: int = 240) -> str | None:
    """Return an SVG of one molecule, or None when RDKit cannot read it."""
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
    return drawer.GetDrawingText()


def assembled_smiles(prefix: str) -> str | None:
    """Read back the part of a SAFE prefix that already forms whole fragments."""
    for candidate in (prefix, prefix.rsplit(".", 1)[0] if "." in prefix else ""):
        if not candidate:
            continue
        try:
            smiles = safe_to_smiles(candidate)
        except Exception:
            smiles = None
        if smiles and Chem.MolFromSmiles(smiles) is not None:
            return smiles
    return None


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_decoder(args.checkpoint, device, use_ema=not args.no_ema)
    tokenizer = load_safe_tokenizer(args.tokenizer)
    token_strings = [
        tokenizer.convert_ids_to_tokens(index) for index in range(len(tokenizer))
    ]
    chemistry_forbidden = tuple(
        sorted(set(isotope_token_ids(token_strings)) | set(foreign_element_token_ids(token_strings)))
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

    metadata = pd.read_csv(args.metadata)
    if args.spec_manifest is not None:
        panel = pd.read_csv(args.spec_manifest, sep="\t")
        wanted = list(panel["spec_name"].astype(str))
        metadata = metadata[metadata["spec_name"].astype(str).isin(wanted)]
    returned = {}
    if args.predictions is not None and args.predictions.exists():
        for line in args.predictions.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                returned[str(row["spec_name"])] = row

    chosen = metadata.drop_duplicates("spec_name").sort_values("neutral_mass")
    step = max(len(chosen) // args.molecules, 1)
    chosen = chosen.iloc[::step].head(args.molecules)

    molecules = []
    for _, record in chosen.iterrows():
        smiles = str(record["smiles"])
        molecule = Chem.MolFromSmiles(smiles)
        safe = smiles_to_safe(smiles)
        if molecule is None or safe is None:
            continue
        encoded = tokenizer(safe, return_tensors="pt")["input_ids"][0]
        fingerprint_bits = AllChem.GetMorganGenerator(
            radius=2, fpSize=4096
        ).GetFingerprint(molecule)
        fingerprint_array = np.zeros(4096, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fingerprint_bits, fingerprint_array)
        fingerprint = torch.from_numpy(fingerprint_array).to(device).unsqueeze(0)
        target_mass = float(conditioning_mass(molecule))
        mass = torch.tensor([target_mass], device=device)

        steps = []
        start = 1
        while start < len(encoded):
            boundary = (
                ((start - 1) // model.config.block_width) + 1
            ) * model.config.block_width + 1
            end = min(boundary, len(encoded))
            input_ids = (
                torch.cat(
                    (
                        encoded[:start],
                        torch.full(
                            (end - start,), tokenizer.mask_token_id, dtype=torch.long
                        ),
                    )
                )
                .to(device)
                .unsqueeze(0)
            )
            with (
                torch.inference_mode(),
                torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ),
            ):
                logits = model.sampling_logits(input_ids, mass, fingerprint)[
                    0, start:end
                ].float()
            probabilities = F.log_softmax(logits, dim=-1).exp()

            for offset in range(end - start):
                position = start + offset
                prefix_ids = encoded[:position].tolist()
                prefix = tokenizer.decode(prefix_ids, skip_special_tokens=True)
                gold_id = int(encoded[position])
                row_probabilities = probabilities[offset]
                ranked = torch.topk(row_probabilities, args.top_k)
                support = full_mask._mass_reachable_token_ids(prefix, target_mass)
                allowed_ids = set(support)
                tokens = []
                for probability, token_id in zip(
                    ranked.values.tolist(), ranked.indices.tolist()
                ):
                    tokens.append(
                        {
                            "token": token_strings[token_id],
                            "probability": round(float(probability), 5),
                            "gold": token_id == gold_id,
                            "verdict": classify(
                                token_id,
                                prefix,
                                allowed_ids,
                                full_mask,
                                syntax_mask,
                                token_strings,
                                target_mass,
                            ),
                        }
                    )
                if gold_id not in ranked.indices.tolist():
                    tokens.append(
                        {
                            "token": token_strings[gold_id],
                            "probability": round(
                                float(row_probabilities[gold_id]), 5
                            ),
                            "gold": True,
                            "verdict": classify(
                                gold_id,
                                prefix,
                                allowed_ids,
                                full_mask,
                                syntax_mask,
                                token_strings,
                                target_mass,
                            ),
                        }
                    )
                written = tokenizer.decode(
                    encoded[: position + 1].tolist(), skip_special_tokens=True
                )
                steps.append(
                    {
                        "position": position,
                        "block": (position - 1) // model.config.block_width,
                        "prefix": prefix,
                        "written": written,
                        "gold_token": token_strings[gold_id],
                        "support": len(support),
                        "support_bits": pack_support(allowed_ids, len(token_strings)),
                        "written_mass": written_mass(written),
                        "tokens": tokens,
                        "assembled": assembled_smiles(written),
                    }
                )
            start = end

        drawn = {}
        for entry in steps:
            smiles_here = entry["assembled"]
            if smiles_here and smiles_here not in drawn:
                drawn[smiles_here] = draw(smiles_here, 300, 210)
        molecules.append(
            {
                "spec_name": str(record["spec_name"]),
                "smiles": smiles,
                "safe": safe,
                "target_mass": target_mass,
                "formula": Chem.rdMolDescriptors.CalcMolFormula(molecule),
                "heavy_atoms": molecule.GetNumHeavyAtoms(),
                "target_svg": draw(smiles, 420, 300),
                # The only chemistry the decoder is given: 4,096 Morgan bits and
                # one mass. Showing it makes the conditioning concrete.
                "fingerprint_bits": pack_support(
                    {index for index, bit in enumerate(fingerprint_array) if bit},
                    len(fingerprint_array),
                ),
                "fingerprint_on": int(fingerprint_array.sum()),
                "vocabulary": len(token_strings),
                "steps": steps,
                "fragment_svgs": drawn,
                "returned": summarise_returned(returned.get(str(record["spec_name"]))),
            }
        )
        print(f"  {record['spec_name']}: {len(steps)} steps, {len(drawn)} fragment drawings")

    payload = {
        "checkpoint": str(args.checkpoint),
        "ppm_tolerance": args.ppm_tolerance,
        "vocabulary_tokens": token_strings,
        "chemistry_forbidden": len(chemistry_forbidden),
        "block_width": int(model.config.block_width),
        "fingerprint_size": 4096,
        "molecules": molecules,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialised = json.dumps(payload)
    args.output.write_text(serialised)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")

    if args.html_template is not None and args.html_output is not None:
        # The page is opened straight off disk, where fetch() of a sibling file is
        # blocked, so the data has to travel inside the document.
        template = args.html_template.read_text()
        marker = "/*DEMO_DATA*/"
        if marker not in template:
            raise ValueError(f"template has no {marker} marker")
        args.html_output.parent.mkdir(parents=True, exist_ok=True)
        args.html_output.write_text(
            template.replace(marker, serialised.replace("</", "<\\/"))
        )
        print(
            f"wrote {args.html_output} "
            f"({args.html_output.stat().st_size / 1e6:.1f} MB)"
        )


def written_mass(prefix: str) -> float | None:
    """Return the mass the written prefix already commits, hydrogens included.

    This is the same lower bound the mass shell prunes against, so the bar on the
    page moves exactly as the constraint sees it move.
    """
    state = _scan(prefix)
    if state is None:
        return None
    return round(float(state.minimum_mass(0.0)), 4)


def pack_support(allowed_ids: set[int], size: int) -> str:
    """Pack the admitted token ids into a base64 bitmap, one bit per token."""
    bits = bytearray((size + 7) // 8)
    for token_id in allowed_ids:
        if 0 <= token_id < size:
            bits[token_id // 8] |= 1 << (token_id % 8)
    return base64.b64encode(bytes(bits)).decode("ascii")


def classify(
    token_id: int,
    prefix: str,
    allowed_ids: set[int],
    full_mask: SafeGrammarMask,
    syntax_mask: SafeGrammarMask,
    token_strings: list[str],
    target_mass: float,
) -> str:
    """Name the first restriction that refuses ``token_id`` after ``prefix``."""
    if token_id in allowed_ids:
        return "allowed"
    if token_id in full_mask.forbidden_token_ids:
        return "chemistry"
    token = token_strings[token_id]
    if token_id not in syntax_mask._valid_token_ids(prefix):
        if "." in token and syntax_mask._valid_token_ids(prefix.rstrip(".")):
            return "connectivity"
        return "connectivity" if "." in token else "syntax"
    return "mass"


def summarise_returned(row: dict | None) -> dict | None:
    """Carry over what the evaluation run actually returned for this spectrum."""
    if row is None:
        return None
    candidates = []
    for candidate in (row.get("candidates") or [])[:3]:
        candidates.append(
            {
                "smiles": candidate["smiles"],
                "svg": draw(candidate["smiles"], 300, 210),
                "mass_error_ppm": candidate.get("mass_error_ppm"),
                "tanimoto": candidate.get("target_fingerprint_tanimoto"),
                "fragments": candidate["smiles"].count(".") + 1,
            }
        )
    return {
        "runtime_seconds": row.get("runtime_seconds"),
        "constraint_dead_ends": row.get("constraint_dead_ends"),
        "attempts": row.get("attempts"),
        "candidates": candidates,
    }


if __name__ == "__main__":
    main()
