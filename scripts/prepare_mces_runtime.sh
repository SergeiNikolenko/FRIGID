#!/usr/bin/env bash
set -euo pipefail

: "${FRIGID_REPO:?Set FRIGID_REPO to a clean worktree}"
: "${BASE_PYTHON:?Set BASE_PYTHON to the existing FRIGID Python executable}"
: "${RUNTIME_DIR:?Set RUNTIME_DIR to a new MCES runtime directory}"
: "${EXPECTED_CODE_COMMIT:?Set EXPECTED_CODE_COMMIT}"

REQUIREMENTS="${REQUIREMENTS:-$FRIGID_REPO/configs/mces_runtime_requirements.txt}"
SITE_PACKAGES="$RUNTIME_DIR/site-packages"
MANIFEST="$RUNTIME_DIR/RUNTIME_MANIFEST.json"

if [[ -e "$RUNTIME_DIR" ]]; then
  echo "MCES runtime directory already exists: $RUNTIME_DIR" >&2
  exit 2
fi
for path in "$BASE_PYTHON" "$REQUIREMENTS"; do
  if [[ ! -e "$path" ]]; then
    echo "Required MCES runtime input does not exist: $path" >&2
    exit 2
  fi
done

actual_commit="$(git -C "$FRIGID_REPO" rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_CODE_COMMIT" ]]; then
  echo "Code commit mismatch: expected $EXPECTED_CODE_COMMIT, got $actual_commit" >&2
  exit 2
fi
if [[ -n "$(git -C "$FRIGID_REPO" status --porcelain --untracked-files=all)" ]]; then
  echo "FRIGID_REPO must be clean: $FRIGID_REPO" >&2
  exit 2
fi

mkdir -p "$SITE_PACKAGES"
uv pip install \
  --target "$SITE_PACKAGES" \
  --no-deps \
  --require-hashes \
  --requirements "$REQUIREMENTS"

export FRIGID_REPO BASE_PYTHON RUNTIME_DIR REQUIREMENTS SITE_PACKAGES MANIFEST
export EXPECTED_CODE_COMMIT
PYTHONPATH="$SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}" "$BASE_PYTHON" - <<'PY'
import hashlib
import importlib
import importlib.metadata
import json
import os
import pathlib
import platform
import socket
import sys
from datetime import datetime, timezone

import joblib
import networkx
import numpy
import pandas
import pulp
import rdkit
import scipy
from myopic_mces import MCES


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


available_solvers = pulp.listSolvers(onlyAvailable=True)
if "PULP_CBC_CMD" not in available_solvers:
    raise RuntimeError(f"PULP_CBC_CMD is unavailable: {available_solvers}")
smoke_result = MCES(
    "CC",
    "CCC",
    solver="PULP_CBC_CMD",
    threshold=15,
    always_stronger_bound=True,
    solver_options={"msg": 0, "timeLimit": 30},
)
if int(smoke_result[1]) != 1:
    raise RuntimeError(f"Unexpected MCES smoke distance: {smoke_result}")

requirements = pathlib.Path(os.environ["REQUIREMENTS"]).resolve()
module_names = ("myopic_mces", "pulp")
overlay_modules = {
    name: str(pathlib.Path(importlib.import_module(name).__file__).resolve())
    for name in module_names
}
site_packages = pathlib.Path(os.environ["SITE_PACKAGES"]).resolve()
if any(site_packages not in pathlib.Path(path).parents for path in overlay_modules.values()):
    raise RuntimeError(f"MCES modules were not loaded from overlay: {overlay_modules}")

manifest = {
    "schema_version": 1,
    "purpose": "frigid_mces_runtime",
    "status": "completed",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "host": socket.gethostname(),
    "code": {
        "repo": str(pathlib.Path(os.environ["FRIGID_REPO"]).resolve()),
        "commit": os.environ["EXPECTED_CODE_COMMIT"],
        "dirty": False,
    },
    "base_python": {
        "path": os.environ["BASE_PYTHON"],
        "resolved_path": str(pathlib.Path(os.environ["BASE_PYTHON"]).resolve()),
        "version": sys.version,
        "platform": platform.platform(),
    },
    "requirements": {
        "path": str(requirements),
        "sha256": sha256_file(requirements),
    },
    "overlay": {
        "site_packages": str(site_packages),
        "versions": {
            "myopic-mces": importlib.metadata.version("myopic-mces"),
            "PuLP": importlib.metadata.version("PuLP"),
        },
        "module_paths": overlay_modules,
    },
    "inherited_versions": {
        "joblib": joblib.__version__,
        "networkx": networkx.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "rdkit": rdkit.__version__,
        "scipy": scipy.__version__,
    },
    "solver": {
        "selected": "PULP_CBC_CMD",
        "available": available_solvers,
        "path": pulp.PULP_CBC_CMD().path,
        "sha256": sha256_file(pathlib.Path(pulp.PULP_CBC_CMD().path)),
    },
    "metric_contract": {
        "threshold": 15,
        "always_stronger_bound": True,
        "solver_time_limit_seconds": 600,
        "implementation": "scripts/multi_compute.py::compute_metrics_for_one",
    },
    "smoke": {
        "smiles_a": "CC",
        "smiles_b": "CCC",
        "expected_distance": 1,
        "result": list(smoke_result),
    },
}
pathlib.Path(os.environ["MANIFEST"]).write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(manifest["overlay"], indent=2))
print(json.dumps(manifest["solver"], indent=2))
print(json.dumps(manifest["smoke"], indent=2))
PY
