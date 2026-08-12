#!/usr/bin/env python3
"""Census the aromatic rings of every gold answer, two ways.

The mask needs a rule for "a closing aromatic ring must be kekulizable", and the
obvious cheap form of it is a list of admissible ring sizes. This script asks the
gold answers whether that form can work, and reports both halves of the answer:

* **Ring perception** -- the size of every all-aromatic ring RDKit finds in the
  gold molecule. This is what a chemist means by an aromatic ring.
* **Ring closure** -- the smallest all-aromatic cycle each ring-closure label
  completes while the string is being written, which is the only thing a mask can
  see. The scan goes through :func:`marlin.grammar._scan`, so it is the state the
  mask itself would read rather than a second parser that could disagree with it.

The two disagree, and the disagreement is the finding: a fused polycycle closes
its perimeter first and the six-membered rings inside it afterwards, so the sizes
a closure completes run far past the sizes a ring ever has. See
``docs/DECODER_PROGRAM.md`` section 5.

Usage:
  PYTHONPATH=src python scripts/audit_marlin_aromatic_rings.py \
      --metadata .../train/metadata.csv --metadata .../val/metadata.csv \
      --metadata .../test/metadata.csv
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

from dlm.utils.utils_chem import smiles_to_safe
from marlin.grammar import _GrammarState, _scan

RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, action="append", required=True)
    parser.add_argument("--smiles-column", default="smiles")
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


def _aromatic_edges(state: _GrammarState) -> set[tuple[int, int]]:
    return {
        bond
        for bond in state.bonds
        if bond[0] in state.aromatic_atoms and bond[1] in state.aromatic_atoms
    }


def _shortest_cycle(edges: set[tuple[int, int]], left: int, right: int) -> int | None:
    """Size of the smallest cycle the edge ``(left, right)`` would close."""
    adjacency: dict[int, list[int]] = collections.defaultdict(list)
    for one, other in edges:
        adjacency[one].append(other)
        adjacency[other].append(one)
    distance = {left: 0}
    queue = collections.deque([left])
    while queue:
        node = queue.popleft()
        if node == right:
            return distance[node] + 1
        for neighbour in adjacency[node]:
            if neighbour not in distance:
                distance[neighbour] = distance[node] + 1
                queue.append(neighbour)
    return None


def closure_cycle_sizes(safe: str) -> list[int | None]:
    """Report what every aromatic ring closure of ``safe`` completes.

    ``None`` marks a closure that joins two aromatic atoms with no aromatic path
    between them, which completes no aromatic cycle at all.
    """
    sizes: list[int | None] = []
    previous = _GrammarState()
    for length in range(1, len(safe) + 1):
        state = _scan(safe[:length])
        if state is None:
            raise ValueError(f"gold answer does not scan at {length}: {safe!r}")
        new_bonds = state.bonds - previous.bonds
        for bond in new_bonds:
            # Only a ring closure can bond two atoms that both already existed.
            if bond[1] > previous.atom_index:
                continue
            if not {bond[0], bond[1]} <= state.aromatic_atoms:
                continue
            sizes.append(_shortest_cycle(_aromatic_edges(previous), *bond))
        previous = state
    return sizes


def main() -> None:
    args = parse_args()
    frames = [pd.read_csv(path) for path in args.metadata]
    table = pd.concat(frames, ignore_index=True)
    if args.limit:
        table = table.iloc[: args.limit]

    perceived = collections.Counter()
    closures = collections.Counter()
    molecules = 0
    failures = 0

    for smiles in table[args.smiles_column]:
        molecule = Chem.MolFromSmiles(str(smiles))
        safe = smiles_to_safe(str(smiles))
        if molecule is None or safe is None:
            failures += 1
            continue
        molecules += 1
        for ring in molecule.GetRingInfo().AtomRings():
            if all(molecule.GetAtomWithIdx(index).GetIsAromatic() for index in ring):
                perceived[len(ring)] += 1
        for size in closure_cycle_sizes(safe):
            closures["none" if size is None else size] += 1

    report = {
        "kind": "MARLIN gold-answer aromatic ring census",
        "git_commit": _git_commit(),
        "metadata": [str(path) for path in args.metadata],
        "molecules": molecules,
        "failures": failures,
        "perceived_aromatic_ring_sizes": {
            str(size): count for size, count in sorted(perceived.items())
        },
        "closure_aromatic_cycle_sizes": {
            str(size): count
            for size, count in sorted(closures.items(), key=lambda item: str(item[0]))
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
