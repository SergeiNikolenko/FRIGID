#!/bin/bash
# Submit the immutable MARLIN validation scorer through Slurm.

set -euo pipefail

CODE=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/code
SHARED_ROOT=/mnt/netstorage/nikolenko/marlin
mkdir -p "$SHARED_ROOT/runs/autoresearch/slurm"
cd "$CODE"
sbatch --wait scripts/slurm_marlin_autoresearch_score.sbatch "$@"
