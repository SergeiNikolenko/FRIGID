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
- This worktree and branch are the strict paper-reproduction lane:
  `/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/paper-worktree`
  on `marlin-paper-reproduction`. Limit implementation, experiments, and
  status reports here to this lane. Do not modify, evaluate, or report the
  Expanding MARLIN / EFlow / EFM lane from this worktree.
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
