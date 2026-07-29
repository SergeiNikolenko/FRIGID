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
