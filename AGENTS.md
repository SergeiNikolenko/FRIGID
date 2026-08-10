# MARLIN Agent Instructions

## Shared storage

- Use `/mnt/netstorage/nikolenko/marlin` as the canonical shared storage root.
- Store large datasets, immutable dataset snapshots, checkpoints, training runs,
  evaluation outputs, and long-lived experiment artifacts under that root.
- Do not duplicate multi-gigabyte artifacts under
  `/home/nikolenko/work/Projects/MARLIN_reproduction_20260717`.
- Keep the Git worktree limited to source code, tests, configuration, and small
  manifests needed to reproduce shared-storage artifacts.
- Slurm jobs must write new checkpoints and run artifacts to the shared storage
  root unless a smoke test explicitly requires a temporary local output.
- Treat live Slurm jobs from this checkout as shared campaign state. Do not
  cancel, hold, release, or replace another MARLIN job merely to acquire the
  GPU; let Slurm serialize them unless the owning run is demonstrably stale or
  failing.
- Treat shared-storage paths as durable user data. Never delete or overwrite
  them without resolving the exact target and obtaining explicit authorization.
- On Spectrum, `/mnt/netstorage/nikolenko/marlin` is the canonical NFS-backed
  root. On the ClearML FARO workers, `/mnt/netstorage` is worker-local storage,
  not the Spectrum filesystem. Never assume files written on one host are
  visible on the other.
- A FARO task may use its local `/mnt/netstorage/nikolenko/marlin` cache only
  after `scripts/materialize_marlin_worker_cache.py` has completed and
  `worker_cache_ready.json` matches the pinned checkpoint and dataset hashes.
  Keep FARO cache bootstrap tasks resumable and record their task ID and exact
  Git commit.

## MARLIN scope

- Implement and evaluate MARLIN from arXiv:2607.04774 directly.
- The strict clean-room reproduction lane may use the official FRIGID
  checkpoint only as the hashed one-time warm start described by the MARLIN
  paper. Do not use the FRIGID sampler or a FRIGID teacher in its objective or
  evaluation.
- A separate user-authorized experimental lane may use the official FRIGID
  checkpoint as a teacher. Label every such run `FRIGID-distilled MARLIN`,
  record checkpoint hashes and provenance, and never present it as the strict
  paper reproduction.
- The user-authorized Expanding MARLIN lane may adapt EFlow/EFM from
  arXiv:2607.21585. Label every run `EFM-inspired MARLIN`, tag it
  `non-paper-architecture`, and keep its configs and checkpoints separate from
  the strict MARLIN reproduction.
- Judge progress with molecular generation metrics (validity, uniqueness,
  candidate return rate, mass validity, and structure similarity), not loss
  alone.

## Evaluation panels

- Do not make decisions on the 32-spectrum micro panel. Its floor is one molecule
  in 32, so `Exact@1` cannot resolve anything below 3.1%, and it swings by up to
  0.12 between neighbouring checkpoints. Several published-then-withdrawn readings
  in `docs/MARLIN_STATUS_REPORT.md` came from reading that noise as signal.
- Evaluate on a full split. `scripts/evaluate_marlin_sharded.sh` shards by spectrum
  across cores; the constrained decoder is CPU bound, not GPU bound, so this is
  close to linear in core count. The 803-spectrum locked test split at 8 candidates
  takes about 1.1 h on 16 shards against about 17 h in one process.
- Pin one BLAS thread per worker (`OMP_NUM_THREADS=1` and friends). Sixteen torch
  processes each defaulting to all 24 cores drove the load average to 86 and ran
  slower than serial.
- Keep the split fixed and treat the candidate budget as the ladder: spectra count
  is what makes a number comparable, budget only moves it along a measurable curve.
  Label every result with its actual budget, never with the budget requested when
  a per-spectrum time cap truncated it.
- Report a metric with its denominator. `Exact@k` averages over all spectra; the
  paper averages Tanimoto and MCES over spectra that produced a candidate, while
  FRIGID's own report averages them over all spectra. Use
  `scripts/score_frigid_convention.py` when comparing against FRIGID's report.
- The periodic evaluation during training should use a full validation split, not
  the micro panel. At about 30 min per checkpoint sharded, a real held-out signal
  is affordable, and without one the recipe has no early stopping and no
  checkpoint selection: the run's own evaluation reported a flat `Exact@1` while a
  matched offline re-evaluation showed candidate return peaking at step 20,000 and
  halving by 50,000.
