#!/bin/bash
# Submit a Slurm job only once the A100 carries no other user's process.
#
# This host IS the Slurm node `spectrum`: `hostname` returns spectrum and
# `scontrol show node spectrum` reports the same 24 cores and the same
# gpu:a100:1. So "the free local A100" and "the idle Slurm gpu partition" are one
# card. Slurm can report the node IDLE while a bare process outside Slurm holds
# the card at 99% utilisation, which is exactly the state this was written in.
# Submitting into that would not evict the other user, but it would halve their
# throughput and ours, so we wait instead.
#
# Usage: submit_when_gpu_idle.sh <sbatch file> [max wait seconds]
set -euo pipefail

SBATCH_FILE="${1:?usage: submit_when_gpu_idle.sh <sbatch file> [max wait seconds]}"
DEADLINE_SECONDS="${2:-86400}"
ME="$(id -un)"
STARTED="$(date +%s)"

foreign_processes() {
    # Empty output means no process from another user holds the card.
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | tr -d ' ' \
        | while read -r pid; do
            [ -n "$pid" ] || continue
            owner="$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')"
            [ -n "$owner" ] || continue
            [ "$owner" = "$ME" ] || echo "$pid $owner"
        done
}

while true; do
    busy="$(foreign_processes)"
    if [ -z "$busy" ]; then
        echo "$(date -Is) gpu clear, submitting $SBATCH_FILE"
        sbatch "$SBATCH_FILE"
        exit 0
    fi
    elapsed=$(( $(date +%s) - STARTED ))
    if [ "$elapsed" -ge "$DEADLINE_SECONDS" ]; then
        echo "$(date -Is) gave up after ${elapsed}s; still held by: $busy"
        exit 1
    fi
    echo "$(date -Is) held by: $(echo "$busy" | tr '\n' ' ') (waited ${elapsed}s)"
    sleep 120
done
