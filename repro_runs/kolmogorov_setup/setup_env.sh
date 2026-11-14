#!/usr/bin/env bash
set -euo pipefail
cd /home/nikolenko/work/Projects/FRIGID
uv venv --python 3.10 .venv
source .venv/bin/activate
uv pip install -U pip setuptools wheel 'Cython<3'
uv pip install -e .
uv pip install 'transformers==4.32.0' ngboost h5py
cd ms-pred
uv pip install --no-build-isolation -e .
cd ..
python - <<'PY'
import torch, transformers, ngboost, h5py
print('python ok')
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('cuda_device_count', torch.cuda.device_count())
print('transformers', transformers.__version__)
PY
