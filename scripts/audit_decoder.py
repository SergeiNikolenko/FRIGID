"""Measure the four defects the decoding page surfaced, over the whole panel."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Descriptors, rdFingerprintGenerator

sys.path.insert(0, "src")
from marlin.token_properties import foreign_element_token_ids, isotope_token_ids  # noqa: E402
from marlin.tokenizer import load_safe_tokenizer  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RT = Path(
    "/mnt/netstorage/nikolenko/marlin/cache/runtime-inputs-spectrum-v1/"
    "16b1af5276034c041e85a4b7c43129a790b4fc091826485b691c93f9f7b699b3"
)
CLEAN = Path("/mnt/netstorage/nikolenko/marlin/evaluations/clean-after/predictions.jsonl")
report: dict[str, object] = {}

# ---------------------------------------------------------------- vocabulary
tokenizer = load_safe_tokenizer(RT / "tokenizer.json")
tokens = [tokenizer.convert_ids_to_tokens(i) for i in range(len(tokenizer))]
forbidden = set(isotope_token_ids(tokens)) | set(foreign_element_token_ids(tokens))
charge = re.compile(r"^\[([A-Za-z][a-z]?)[Hh]?\d*([+-])(\d*)\]$")

odd_charges, hydrogen_tokens, allowed_odd = [], [], []
for index, token in enumerate(tokens):
    match = charge.match(token)
    if match:
        magnitude = int(match.group(3) or 1)
        if magnitude >= 2:
            odd_charges.append(token)
            if index not in forbidden:
                allowed_odd.append(token)
    if re.match(r"^\[[0-9]*H[^\]]*\]$", token):
        hydrogen_tokens.append(token)

report["vocabulary"] = {
    "size": len(tokens),
    "withheld_by_chemistry": len(forbidden),
    "charge_two_or_more": len(odd_charges),
    "charge_two_or_more_still_allowed": len(allowed_odd),
    "examples_allowed": sorted(allowed_odd)[:20],
    "hydrogen_tokens": len(hydrogen_tokens),
    "hydrogen_examples": sorted(hydrogen_tokens)[:20],
}

# ------------------------------------------------------- conditioning quality
dreams = np.load(RT / "val/dreams_predictions.npz")
probabilities = dreams["probs"]
names = [str(name) for name in dreams["spectrum_ids"]]
metadata = pd.read_csv(RT / "val/metadata.csv")
row_of = {name: index for index, name in enumerate(names)}
generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)

recalls, precisions, real_counts, predicted_counts = [], [], [], []
for _, record in metadata.iterrows():
    spec = str(record["spec_name"])
    if spec not in row_of:
        continue
    molecule = Chem.MolFromSmiles(str(record["smiles"]))
    if molecule is None:
        continue
    real = set(generator.GetFingerprint(molecule).GetOnBits())
    predicted = set(np.nonzero(probabilities[row_of[spec]] >= 0.95)[0].tolist())
    if not real:
        continue
    hit = len(real & predicted)
    recalls.append(hit / len(real))
    precisions.append(hit / max(len(predicted), 1))
    real_counts.append(len(real))
    predicted_counts.append(len(predicted))

report["conditioning"] = {
    "spectra": len(recalls),
    "median_recall": round(float(np.median(recalls)), 3),
    "median_precision": round(float(np.median(precisions)), 3),
    "mean_real_bits": round(float(np.mean(real_counts)), 1),
    "mean_predicted_bits": round(float(np.mean(predicted_counts)), 1),
    "spectra_with_recall_below_0.5": int(sum(1 for value in recalls if value < 0.5)),
}

# ------------------------------------------------------------ returned candidates
if CLEAN.exists():
    rows = [json.loads(line) for line in CLEAN.read_text().splitlines() if line.strip()]
    fragmented, hydrogen_padded, total_candidates = 0, 0, 0
    dead_end_only, returned_any = 0, 0
    fragment_sizes = Counter()
    for row in rows:
        candidates = row.get("candidates", [])
        returned_any += bool(candidates)
        dead_end_only += not candidates
        for candidate in candidates:
            total_candidates += 1
            molecule = Chem.MolFromSmiles(candidate["smiles"])
            if molecule is None:
                continue
            pieces = Chem.GetMolFrags(molecule, asMols=True)
            fragment_sizes[len(pieces)] += 1
            if len(pieces) > 1:
                fragmented += 1
                light = sum(
                    Descriptors.ExactMolWt(piece)
                    for piece in pieces
                    if piece.GetNumHeavyAtoms() <= 1
                )
                if light > 0:
                    hydrogen_padded += 1
    report["returned"] = {
        "spectra": len(rows),
        "spectra_with_a_candidate": returned_any,
        "spectra_where_every_attempt_died": dead_end_only,
        "candidates": total_candidates,
        "candidates_in_pieces": fragmented,
        "candidates_padded_with_one_atom_fragments": hydrogen_padded,
        "fragment_count_histogram": dict(sorted(fragment_sizes.items())[:8]),
        "mean_attempts": round(float(np.mean([r["attempts"] for r in rows])), 1),
        "mean_dead_ends": round(float(np.mean([r["constraint_dead_ends"] for r in rows])), 1),
        "mean_runtime_seconds": round(float(np.mean([r["runtime_seconds"] for r in rows])), 1),
        "truncated_by_time_budget": int(sum(1 for r in rows if r.get("truncated"))),
    }

print(json.dumps(report, indent=1))
