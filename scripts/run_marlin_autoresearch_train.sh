#!/bin/bash
# Submit a committed MARLIN candidate from an isolated detached worktree.

set -euo pipefail

REPRO_ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$REPRO_ROOT/code"
COMMIT="$(git -C "$CODE" rev-parse HEAD)"
WORKTREE_ROOT="$REPRO_ROOT/autoresearch-worktrees"
WORKTREE="$WORKTREE_ROOT/train-${COMMIT:0:12}"
mkdir -p "$WORKTREE_ROOT"

exec 9>"$WORKTREE_ROOT/.worktree.lock"
flock 9
if [[ ! -e "$WORKTREE/.git" ]]; then
    git -C "$CODE" worktree add --detach "$WORKTREE" "$COMMIT"
fi
WORKTREE_COMMIT="$(git -C "$WORKTREE" rev-parse HEAD)"
if [[ "$WORKTREE_COMMIT" != "$COMMIT" ]]; then
    echo "Training worktree commit mismatch: $WORKTREE_COMMIT != $COMMIT" >&2
    exit 2
fi
flock -u 9

cd "$WORKTREE"
sbatch \
    --export="ALL,MARLIN_CODE_ROOT=$WORKTREE,MARLIN_REPRO_ROOT=$REPRO_ROOT" \
    scripts/slurm_marlin_train.sbatch
