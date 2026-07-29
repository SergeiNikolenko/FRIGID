#!/bin/bash
# Submit the immutable MARLIN validation scorer through Slurm.

set -euo pipefail

REPRO_ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$REPRO_ROOT/code"
SHARED_ROOT=/mnt/netstorage/nikolenko/marlin
# Program revision 1: never let a candidate change its evaluator.
COMMIT=8082c57050b65d4be87f4a403fb97338ce307871
WORKTREE_ROOT="$REPRO_ROOT/autoresearch-worktrees"
WORKTREE="$WORKTREE_ROOT/scorer-${COMMIT:0:12}"
mkdir -p "$SHARED_ROOT/runs/autoresearch/slurm"
mkdir -p "$WORKTREE_ROOT"

exec 9>"$WORKTREE_ROOT/.worktree.lock"
flock 9
if [[ ! -e "$WORKTREE/.git" ]]; then
    git -C "$CODE" worktree add --detach "$WORKTREE" "$COMMIT"
fi
WORKTREE_COMMIT="$(git -C "$WORKTREE" rev-parse HEAD)"
if [[ "$WORKTREE_COMMIT" != "$COMMIT" ]]; then
    echo "Scorer worktree commit mismatch: $WORKTREE_COMMIT != $COMMIT" >&2
    exit 2
fi
flock -u 9

ARGS=()
EXPECT_OUTPUT=0
for argument in "$@"; do
    if [[ "$EXPECT_OUTPUT" == 1 ]]; then
        if [[ "$argument" != /* ]]; then
            argument="$CODE/$argument"
        fi
        EXPECT_OUTPUT=0
    elif [[ "$argument" == "--output" ]]; then
        EXPECT_OUTPUT=1
    fi
    ARGS+=("$argument")
done
if [[ "$EXPECT_OUTPUT" == 1 ]]; then
    echo "--output requires a path" >&2
    exit 2
fi

cd "$WORKTREE"
sbatch \
    --wait \
    --export="ALL,MARLIN_CODE_ROOT=$WORKTREE,MARLIN_REPRO_ROOT=$REPRO_ROOT" \
    scripts/slurm_marlin_autoresearch_score.sbatch \
    "${ARGS[@]}"
