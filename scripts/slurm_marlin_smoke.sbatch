#!/bin/bash
#SBATCH --job-name=marlin-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/smoke-%j.out

set -euo pipefail

ROOT=/home/nikolenko/work/Projects/MARLIN_reproduction_20260717
CODE="$ROOT/code"
PYTHON=$ROOT/.venv/bin/python
export PYTHONPATH="$CODE/src:$ROOT/vendor"

"$PYTHON" - <<'PY'
import json
from pathlib import Path

import torch

from marlin.model import MarlinDecoder, MarlinDecoderConfig

config = MarlinDecoderConfig()
model = MarlinDecoder(config).cuda().to(torch.bfloat16)
tokens = torch.full((1, 16), 4, dtype=torch.long, device="cuda")
tokens[:, 0] = 0
mass = torch.tensor([250.0], device="cuda")
fingerprint = torch.zeros((1, 4096), device="cuda")
fingerprint[:, [5, 42, 500, 1000]] = 1
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    logits = model(tokens, mass, fingerprint)
result = {
    "cuda": torch.cuda.get_device_name(),
    "parameters": sum(parameter.numel() for parameter in model.parameters()),
    "shape": list(logits.shape),
    "finite": bool(torch.isfinite(logits).all()),
    "peak_memory_bytes": torch.cuda.max_memory_allocated(),
}
output = Path("/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/manifests/smoke.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY
