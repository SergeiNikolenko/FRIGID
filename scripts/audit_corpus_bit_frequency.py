#!/usr/bin/env python
"""Measure the Morgan bit frequency of the fp2mol corpus.

The error model bins bit indices by how often they fire in the corpus the
training data will come from, so that number has to be measured on fp2mol and
not on the 6,748-molecule adaptation set. Stereochemistry is stripped first,
because the stream strips it.

Measured 2026-08-13 over 16,777,216 molecules (16 row groups, one from each of
16 shards, 16 processes): 271.7 s wall, 61,741 molecules/s aggregate,
3,859 molecules/s/core, mean 50.28 on-bits, all 4,096 bit indices used,
791 bits above p=0.01 and 62 above p=0.1.
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from marlin.corpus_stream import FP2MOL_SNAPSHOT, corpus_row_groups

RDLogger.DisableLog("rdApp.*")


def _count(task):
    path, row_group, bits, remove_stereo = task
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=bits)
    table = pq.ParquetFile(path).read_row_group(row_group, columns=["smiles"])
    frequency = np.zeros(bits, dtype=np.int64)
    rows = 0
    on_bits = 0
    for smiles in table.column("smiles").to_pylist():
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            continue
        if remove_stereo:
            Chem.RemoveStereochemistry(molecule)
        vector = generator.GetFingerprintAsNumPy(molecule)
        frequency += vector
        on_bits += int(vector.sum())
        rows += 1
    return frequency, rows, on_bits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=FP2MOL_SNAPSHOT)
    parser.add_argument("--row-groups", type=int, default=16)
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument("--bits", type=int, default=4096)
    parser.add_argument("--keep-stereo", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    refs = corpus_row_groups(arguments.snapshot)
    stride = max(len(refs) // arguments.row_groups, 1)
    chosen = refs[::stride][: arguments.row_groups]
    tasks = [
        (ref.path, ref.row_group, arguments.bits, not arguments.keep_stereo)
        for ref in chosen
    ]

    started = time.time()
    with Pool(arguments.processes) as pool:
        results = pool.map(_count, tasks)
    elapsed = time.time() - started

    frequency = sum(result[0] for result in results)
    rows = sum(result[1] for result in results)
    on_bits = sum(result[2] for result in results)
    print(
        f"molecules={rows:,} in {elapsed:.1f}s on {arguments.processes} processes "
        f"= {rows / elapsed:,.0f} mol/s aggregate, "
        f"{rows / elapsed / arguments.processes:,.0f} mol/s/core"
    )
    print(f"mean on-bits={on_bits / max(rows, 1):.2f}, unused bit indices={int((frequency == 0).sum())}")
    np.savez(arguments.output, freq=frequency, n=rows, row_groups=len(chosen))
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
