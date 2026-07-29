#!/bin/bash

set -euo pipefail

SHARED_ROOT="${MARLIN_SHARED_ROOT:-/mnt/netstorage/nikolenko/marlin}"
FRIGID_CHECKPOINT="$SHARED_ROOT/checkpoints/frigid/DLM.ckpt"
SNAPSHOT_MANIFEST="$SHARED_ROOT/safe-gpt-16d0be9ad6177ae683a32a86204530e8ee624a0f/manifest.json"
HOST_MOUNT_ROOT="${MARLIN_HOST_MOUNT_ROOT:-}"
DISCOVER_STORAGE="${MARLIN_DISCOVER_STORAGE:-false}"

printf 'git_commit=%s\n' "$(git rev-parse HEAD)"
printf 'hostname=%s\n' "$(hostname)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader

if [[ -n "$HOST_MOUNT_ROOT" ]]; then
    for relative_path in netstorage ligandpro/shared_storage shared_storage; do
        candidate="$HOST_MOUNT_ROOT/$relative_path"
        if [[ -e "$candidate" ]]; then
            printf 'host_storage_candidate=present path=%s\n' "$candidate"
        else
            printf 'host_storage_candidate=absent path=%s\n' "$candidate"
        fi
    done

    if [[ "$DISCOVER_STORAGE" == "true" && -d "$HOST_MOUNT_ROOT/netstorage" ]]; then
        find "$HOST_MOUNT_ROOT/netstorage" -mindepth 1 -maxdepth 2 -type d \
            -printf 'host_storage_directory=%p\n' | sort | head -n 100
    fi
fi

for path in "$SHARED_ROOT" "$FRIGID_CHECKPOINT" "$SNAPSHOT_MANIFEST"; do
    if [[ ! -r "$path" ]]; then
        printf 'required_path_unreadable=%s\n' "$path" >&2
        exit 2
    fi
    stat --printf='readable_path=%n bytes=%s\n' "$path"
done

if [[ -d /mnt/ligandpro/shared_storage ]]; then
    printf 'ligandpro_shared_storage=present\n'
else
    printf 'ligandpro_shared_storage=absent\n'
fi

python3 - <<'PY'
import importlib.util
import platform

print(f"python={platform.python_version()}")
for package in ("clearml", "torch"):
    print(f"{package}_available={importlib.util.find_spec(package) is not None}")
PY
