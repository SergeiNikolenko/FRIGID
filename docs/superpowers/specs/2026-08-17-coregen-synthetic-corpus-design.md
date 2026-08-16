# Synthetic spectral corpus for CoRe-Gen encoder pretraining — design

Status: proposed, not approved for the full run.
Date: 2026-08-17.
Scope: sub-project 1 of 5 in the CoRe-Gen (arXiv 2605.12980) reproduction.

## 1. Why this exists

CoRe-Gen's largest single component is synthetic-spectrum pretraining of the encoder: its
own ablation prices removal at **-8.08 pp** (19.54% -> 11.46% Top-1 on NPLIB1), larger than
the frequency-aware corruption this project already implements (-4.77 pp). The paper builds
it from "approximately 790K valid organic molecules", each paired with five in-silico MS/MS
spectra: CFM-ID 4.0 at 10, 20 and 40 eV, plus ICEBERG and SCARF.

No such corpus exists here, and no encoder training loop exists here either. This document
covers only the corpus. The encoder is sub-project 2.

**This sub-project reverses a standing project constraint.** `docs/DECODER_PROGRAM.md:7-16`
states "Encoders are taken as given ... we do not train or redesign an encoder". Reproducing
CoRe-Gen in full requires training one. That reversal is a deliberate decision recorded here,
not an oversight; §1 of `DECODER_PROGRAM.md` must be amended when this lands.

## 2. Measured cost, not estimated

All numbers below were measured on this host on 2026-08-17 against the pinned corpus, using
`wishartlab/cfmid:latest` (reports itself as CFM-ID 4.4.7) under Docker.

| Quantity | Measured value | How |
|---|---|---|
| Single-process cost | **11.16 s/molecule** | 10 corpus molecules, one `cfm-predict`, 1m51.6s wall |
| 20-process throughput | **119 molecules / 105 s** | 20 shards x 6 molecules, 120 total |
| Effective per-core cost | **~17.7 s/molecule** | derived from the above; ~1.6x single-process, memory contention |
| Output size | **11.2 kB/molecule** | 112 kB for 10 molecules |
| Energies per invocation | **3** (`energy0`, `energy1`, `energy2`) | one `cfm-predict` call emits 10/20/40 eV together |

Consequences:

- 790K molecules on 16 cores cost **~242 hours (~10 days)** of continuous CPU. On 20 cores,
  ~194 hours (~8 days). This is the dominant cost of the entire reproduction.
- Output for 790K molecules is **~8.8 GB**, which fits `/mnt/netstorage` (503 GB free) but not
  the root filesystem (39 GB free, 96% used). The corpus must be written to netstorage.
- CFM-ID emits all three collision energies in one invocation, so three of the paper's five
  spectrum types cost one pass, not three.

## 3. The pathological tail, and what it forces

One molecule of the 120 (`169615332`, a cardiolipin-like lipid: MW 1546.0, 108 heavy atoms,
74 rotatable bonds) had **not finished after 11m38s** and was killed, against a median
completion of ~9 s. CFM-ID enumerates fragments combinatorially, so cost grows sharply with
rotatable-bond count and heavy-atom count rather than smoothly with mass.

A single such molecule occupies a core for longer than 200 ordinary ones. At 790K molecules,
even a 0.8% incidence at that cost would dominate the wall clock. Two mechanisms are therefore
mandatory, not optional:

1. **Admission filter before simulation** — reject molecules outside the drug-like/natural-
   product envelope the evaluation axis actually contains. Proposed gate: heavy atoms <= 60
   and MW <= 900 Da, applied at selection time and recorded as a named rejection reason.
2. **Per-molecule timeout inside the runner** — a molecule that exceeds the timeout is
   abandoned, counted under a named reason, and never retried. The corpus is a sample, not a
   census; losing the tail is correct, silently hanging on it is not.

Both counts must be published in the manifest. A run that drops 4% of its input and does not
say so reads as complete coverage when it is not.

## 4. Leakage: the failure that would invalidate everything

The corpus feeds an encoder that is scored on NPLIB1. `src/marlin/corpus_stream.py:144-151`
already refuses to start without an exclusion list, and `configs/marlin_nplib1.yaml` pins
`data/nplib1_holdout_inchikeys_v2.csv` (1,095 connectivity blocks, sha256
`7d1f4593...5dab44a`). The selection step here **must apply the same exclusion list and the
same connectivity-key rule** (first InChIKey block), and must refuse to run if the list is
missing or excludes nothing.

This is stricter than the paper, which removes only test overlap. When the resulting number is
compared to 19.54%, that difference must be stated, as `docs/DECODER_PROGRAM.md:997` already
requires for the decoder corpus.

## 5. Components and boundaries

Four units, each independently testable:

**5.1 Selector** (`scripts/select_synthetic_corpus_molecules.py`)
Streams the pinned `safe-gpt` snapshot (933,382,869 rows, 94 shards, sha256-pinned), applies
the admission filter and the holdout exclusion, deduplicates by connectivity key, and emits a
molecule list with a fixed seed. Input: snapshot path, exclusion CSV, target count, seed.
Output: a parquet/TSV of `(mol_id, smiles, inchikey_block)` plus a rejection tally by reason.
Deterministic: the same seed and same snapshot must yield a byte-identical list.

**5.2 CFM-ID runner** (`scripts/run_cfm_id_shard.py` + sbatch wrapper)
Takes a shard of the molecule list, runs `cfm-predict` inside the Docker image with a
per-molecule timeout, and writes one `.log` per molecule. Idempotent and resumable: a shard
re-run skips molecules whose output already exists and validates. Records per-molecule wall
time so §2's numbers can be re-derived from the real run rather than trusted from this spike.

Note: container output is owned by root. The runner must either run the container with
`--user $(id -u):$(id -g)` or chown afterwards; the spike hit this and needed sudo to clean up.

**5.3 Neural simulators** (`scripts/run_iceberg_scarf.py`)
ICEBERG (`ms_pred/dag_pred`) and SCARF (`ms_pred/scarf_pred`) are vendored in `code/ms-pred/`
but **no weights are present on disk**. This unit is blocked until weights are obtained.
Two options, to be decided in sub-project 1b: download the ms-pred authors' MassSpecGym-trained
weights (README lists a Dropbox link; these are trained on less-curated data than their NIST
versions and will not match the paper), or train ICEBERG/SCARF here on CANOPUS
(`data/mist_cf_official/canopus_train/`, present). The second is a sub-project of its own.

**5.4 Packer** (`scripts/pack_synthetic_corpus.py`)
Converts per-molecule `.log` files into sharded parquet with the same pinning discipline the
decoder corpus already uses: `manifest.json` with per-shard sha256, file list sha256, row
counts, generator versions (CFM-ID image digest, ms-pred commit), and the rejection tallies.
Consumers verify hashes before reading.

## 6. Walking skeleton first

The full 790K run is ~10 days of 16 cores. Committing to that before the pipeline has produced
a single validated end-to-end artifact would be the expensive kind of mistake.

Therefore: **pilot at 20K molecules, CFM-ID only** (~6 hours on 16 cores, ~220 MB). The pilot
exercises selection, CFM-ID simulation, packing, hash verification and the resume path, and
produces a corpus large enough to smoke-test the sub-project 2 encoder loop. It deliberately
excludes ICEBERG and SCARF, whose weights do not exist on disk (§5.3) — that dependency must
not block the first end-to-end artifact.

Only after the pilot round-trips does the full run get launched, and its size is then a budget
decision with real numbers attached rather than an extrapolation from 120 molecules.

## 7. Testing

- Selector: exclusion list actually excludes (a known NPLIB1 InChIKey block must not survive);
  empty or missing list refuses to run; same seed yields identical output; admission filter
  rejects the cardiolipin case by name.
- Runner: a molecule exceeding the timeout is recorded as timed out and does not hang the
  shard; re-running a completed shard performs no work; output files are owned by the invoking
  user, not root.
- Packer: a corrupted shard fails hash verification rather than being silently read; the
  manifest's row count matches the packed rows; rejection tallies sum to inputs minus outputs.
- End-to-end on ~200 molecules, asserted in CI-time (not GPU-time) as part of the pilot.

## 8. Open questions for sub-project 1b

1. ICEBERG/SCARF weights: download mismatched public weights, or train on CANOPUS?
2. Does the encoder actually need all five spectrum types, or do the three CFM-ID energies
   carry most of the benefit? The paper does not ablate the simulators against each other, so
   this is unmeasured — and it is worth measuring here before paying for the neural simulators,
   because if CFM-ID alone suffices, sub-project 1b disappears.
3. Final corpus size, as a budget decision after the pilot.
