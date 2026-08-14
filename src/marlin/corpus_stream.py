"""Stream fp2mol as (corrupted fingerprint, SAFE) training examples.

The corpus is the ``datamol-io/safe-gpt`` snapshot pinned at
``/mnt/netstorage/nikolenko/marlin/safe-gpt-16d0be9ad6177ae683a32a86204530e8ee624a0f``:
94 parquet shards, 934 row groups, **933,382,869 rows**, 67 GB, columns
``mol_id / smiles / safe / source``. ``manifest.json`` carries a sha256 per
shard, so a run can pin the exact bytes it read.

Two facts decide the design.

* The corpus ships a SAFE string, but it is a *stereo* SAFE string: 68.6% of the
  smiles carry ``@``, while **0 of 6,748** NPLIB1 adaptation molecules do, and
  the released decoder was pretrained under ``remove_stereo: True``
  (``configs/fp2mol_pretraining.yaml:26``). Reusing the shipped SAFE column
  would train the decoder on a token distribution the adaptation set and the
  evaluation panels never contain. So the stereo is stripped and the SAFE string
  is rebuilt, at a measured 553 molecules/s/core against 5,820 for the shipped
  column.
* Reading the parquet is free by comparison: a 1,048,576-row group decodes in
  0.27-0.39 s off the NFS mount, i.e. 2.7-3.9 million rows/s, which is four
  orders of magnitude above what a training step consumes.

The corrupted fingerprint is *not* produced here. It is produced on the training
device by :class:`marlin.encoder_error_model.EncoderErrorModel`, which is a pair
of tensor ops on ``[batch, 4096]`` - keeping it there means the corruption seed
lives with the training step, the same true fingerprint is corrupted differently
on every replay, and a CPU worker never becomes the bottleneck.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from dlm.utils.utils_chem import safe_to_smiles, smiles_to_safe

RDLogger.DisableLog("rdApp.*")

FP2MOL_SNAPSHOT = Path(
    "/mnt/netstorage/nikolenko/marlin/"
    "safe-gpt-16d0be9ad6177ae683a32a86204530e8ee624a0f"
)
FINGERPRINT_BITS = 4096

# The exclusion list travels with the code. It used to be an absolute path
# outside the repository (``configs/marlin_nplib1.yaml``), which a worker that
# receives its code by ``git clone`` does not have --- and a missing exclusion
# list is not a crash, it is a run that trains on the panel it is scored on.
# 1,095 connectivity blocks, sha256
# 7d1f45937f284dbc9dc93be0ff6ae6eedf02b1cc293496acff7ffcd8c5dab44a.
PACKAGED_HOLDOUT_INCHIKEYS = (
    Path(__file__).resolve().parents[2] / "data" / "nplib1_holdout_inchikeys_v2.csv"
)
PACKAGED_HOLDOUT_INCHIKEYS_SHA256 = (
    "7d1f45937f284dbc9dc93be0ff6ae6eedf02b1cc293496acff7ffcd8c5dab44a"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def resolve_repository_path(path: str | Path) -> Path:
    """Resolve a repo-relative path against the repository, not the cwd.

    A configuration that names ``data/nplib1_holdout_inchikeys_v2.csv`` has to
    mean the same file wherever the process was started from, or the exclusion
    list is present on the machine that wrote the config and missing on the one
    that runs it.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPOSITORY_ROOT / candidate


@dataclass(frozen=True)
class RowGroupRef:
    """One unit of work: a shard path and a row-group index inside it."""

    path: str
    row_group: int


def corpus_row_groups(snapshot: str | Path = FP2MOL_SNAPSHOT) -> list[RowGroupRef]:
    """Enumerate every row group without reading a single data page."""
    snapshot = Path(snapshot)
    manifest = json.loads((snapshot / "manifest.json").read_text())
    refs: list[RowGroupRef] = []
    for entry in manifest["files"]:
        path = snapshot / entry["path"]
        groups = pq.ParquetFile(path).metadata.num_row_groups
        refs.extend(RowGroupRef(str(path), index) for index in range(groups))
    return refs


def _unit_hash(payload: str) -> float:
    digest = hashlib.sha256(payload.encode()).digest()[:8]
    return int.from_bytes(digest, "big") / 2**64


def _stable_seed(payload: str) -> int:
    """A 32-bit seed that does not depend on the interpreter's hash salt."""
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")


class Fp2MolStream(torch.utils.data.IterableDataset):
    """Endless, worker-sharded, deterministic stream of corpus molecules.

    Each yielded example is the dict shape ``MarlinCollator`` already accepts:
    ``{"safe", "fingerprint", "precursor_mass"}``. The fingerprint is the
    molecule's **true** Morgan vector; corruption happens on device.

    Determinism is by construction rather than by luck: the row-group order is a
    hash permutation of ``(seed, epoch, ref)``, so two workers never collide and
    a resumed run replays the same order.
    """

    def __init__(
        self,
        *,
        snapshot: str | Path = FP2MOL_SNAPSHOT,
        tokenizer,
        max_length: int = 256,
        fingerprint_bits: int = FINGERPRINT_BITS,
        exclude_inchikeys: str | Path | None = None,
        allow_evaluation_structures: bool = False,
        remove_stereo: bool = True,
        seed: int = 0,
        row_groups: list[RowGroupRef] | None = None,
        limit: int | None = None,
    ) -> None:
        super().__init__()
        # Measured on ten row groups: the first 10,000,000 corpus molecules --- the
        # size of one paired stage-1 experiment --- carry 74 of the 701 locked-test
        # connectivity blocks and 23 of the 320 clean-panel ones. A stream that
        # defaults to no exclusion therefore trains on the panel it is about to be
        # scored on, silently. Saying "yes, unfiltered" has to be an act.
        if exclude_inchikeys is None and not allow_evaluation_structures:
            raise ValueError(
                "Fp2MolStream needs exclude_inchikeys; the corpus contains evaluation "
                "structures (23 of the 320 clean-panel blocks in the first 10M rows). "
                "Pass data/nplib1_holdout_inchikeys_v2.csv, or "
                "allow_evaluation_structures=True to train on them deliberately."
            )
        self.snapshot = Path(snapshot)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.fingerprint_bits = int(fingerprint_bits)
        self.remove_stereo = bool(remove_stereo)
        self.seed = int(seed)
        self.limit = limit
        self.row_groups = row_groups if row_groups is not None else corpus_row_groups(snapshot)
        if not self.row_groups:
            raise ValueError(f"no parquet row groups under {self.snapshot}")
        self.excluded = _load_excluded(exclude_inchikeys)
        if exclude_inchikeys is not None and not self.excluded:
            # An empty list reads as "filtered" everywhere downstream while
            # excluding nothing, which is the same failure wearing a file name.
            raise ValueError(f"{exclude_inchikeys} lists no connectivity blocks")
        self.rejections: dict[str, int] = {}

    # ------------------------------------------------------------------
    def _shard(self) -> list[RowGroupRef]:
        info = torch.utils.data.get_worker_info()
        if info is None:
            return list(self.row_groups)
        return [
            ref
            for index, ref in enumerate(self.row_groups)
            if index % info.num_workers == info.id
        ]

    def __iter__(self) -> Iterator[dict]:
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=self.fingerprint_bits
        )
        shard = self._shard()
        emitted = 0
        epoch = 0
        while True:
            order = sorted(
                shard,
                key=lambda ref: _unit_hash(f"{self.seed}:{epoch}:{ref.path}:{ref.row_group}"),
            )
            emitted_this_epoch = 0
            for ref in order:
                table = pq.ParquetFile(ref.path).read_row_group(
                    ref.row_group, columns=["smiles"]
                )
                smiles_list = table.column("smiles").to_pylist()
                # Shuffling row groups is not enough. A row group is a million
                # consecutive corpus rows, and the corpus is written in source
                # order, so without this a batch is a million neighbours from
                # one library rather than a sample of the corpus.
                # Not ``hash()``: Python salts string hashing per process, so a
                # dataloader worker would draw a different order than a replay.
                permutation = np.random.default_rng(
                    _stable_seed(f"{self.seed}:{epoch}:{ref.path}:{ref.row_group}")
                ).permutation(len(smiles_list))
                for index in permutation:
                    example = self._build(smiles_list[index], generator)
                    if example is None:
                        continue
                    yield example
                    emitted += 1
                    emitted_this_epoch += 1
                    if self.limit is not None and emitted >= self.limit:
                        return
            if emitted_this_epoch == 0:
                # An endless stream over a corpus that yields nothing is a hang,
                # not a stream: it would sit in the dataloader burning a worker
                # while the trainer waits on a batch that can never arrive.
                raise ValueError(
                    "the corpus stream completed a full pass and emitted nothing; "
                    f"rejections={self.rejections}"
                )
            epoch += 1

    # ------------------------------------------------------------------
    def _reject(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1

    def _build(self, smiles: str, generator) -> dict | None:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            self._reject("unparsable_smiles")
            return None
        if self.remove_stereo:
            Chem.RemoveStereochemistry(molecule)
        canonical = Chem.MolToSmiles(molecule)
        safe = smiles_to_safe(canonical)
        if not safe:
            self._reject("no_safe_encoding")
            return None
        if len(self.tokenizer.encode(safe, add_special_tokens=True)) > self.max_length:
            self._reject("too_long")
            return None
        # Encoding to SAFE is not the contract; the collator decodes the SAFE
        # string back with ``fix=False`` and raises on anything RDKit refuses.
        # A corpus molecule can encode cleanly and still fail that round trip —
        # the same class of string the decoder itself writes and RDKit rejects,
        # an aromatic atom outside a ring or a ring with no Kekule structure.
        # Job 795 died 6:39 in on exactly this, so the stream must apply the
        # collator's own test rather than a weaker one.
        decoded = safe_to_smiles(safe, fix=False)
        if not decoded or Chem.MolFromSmiles(decoded) is None:
            self._reject("no_safe_round_trip")
            return None
        if self.excluded:
            key = Chem.MolToInchiKey(molecule).split("-")[0]
            if key in self.excluded:
                self._reject("evaluation_structure")
                return None
        bits = generator.GetFingerprintAsNumPy(molecule).astype(np.float32)
        return {
            "safe": safe,
            "fingerprint": bits,
            "precursor_mass": float(Descriptors.ExactMolWt(molecule)),
        }


def _load_excluded(path: str | Path | None) -> frozenset[str]:
    if path is None:
        return frozenset()
    import pandas as pd

    table = pd.read_csv(path)
    column = "inchi" if "inchi" in table else "inchikey"
    return frozenset(str(value).split("-")[0] for value in table[column].dropna())
