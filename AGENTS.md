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
- Treat shared-storage paths as durable user data. Never delete or overwrite
  them without resolving the exact target and obtaining explicit authorization.

## MARLIN scope

- Implement and evaluate MARLIN from arXiv:2607.04774 directly.
- Do not use FRIGID code, weights, samplers, warm starts, or architectural
  components in MARLIN training or evaluation.
- Judge progress with molecular generation metrics (validity, uniqueness,
  candidate return rate, mass validity, and structure similarity), not loss
  alone.

