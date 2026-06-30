# FRIGID DLM Adaptation To MIST Fingerprints

Date: 2026-06-30

## Objective

Fine-tune DLM on the fingerprint distribution that MIST actually emits. This is
the selected next experiment after direct DreaMS replacement attempts failed to
approach the MIST fingerprint baseline.

The goal is to reduce the paired DLM robustness gap between clean
`ground_truth` Morgan fingerprints and realistic `mist_binary` fingerprints.

## Inputs

- Host: `spectrum`
- Code checkout:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head`
- Canonical run directory:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dlm_mist_fingerprint_adaptation_20260630T050906Z`
- tmux session: `frigid_dlm_mist_adapt`
- Code commit: `44a5ff0`
- DLM checkpoint:
  `/home/nikolenko/work/Projects/FRIGID/repro_cache/DLM.ckpt`
- MIST checkpoint:
  `/home/nikolenko/work/Projects/FRIGID/repro_cache/mist_msg.pt`
- MSG data:
  `/home/nikolenko/work/Projects/FRIGID/repro_cache/msg`

The run reuses the existing full train-split MIST export:

```text
/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/mist_dreams_residual_20260629T135651Z/mist_train
```

Export summary:

- Split: train.
- Rows: 191,216.
- Fingerprint bits: 4096.
- Training key: `mist_binary`.
- Available arrays: `probs`, `mist_probs`, `mist_binary`, `ground_truth`.
- Export threshold: 0.187.

The canonical run directory symlinks the large export files instead of copying
the 5.5 GB NPZ payload.

## Code Fixes Needed Before Launch

The adaptation config and training path needed two launch fixes:

1. `configs/fp2mol_finetune_mist_fingerprints.yaml` was updated to match the
   real `DLM.ckpt` architecture:
   - hidden size: 896;
   - intermediate size: 3584;
   - attention heads: 14;
   - independent fingerprint cross-attention (`use_shared_cross_attention:
     False`);
   - global batch size reduced to 256 after batch 512 exceeded A100 80GB memory.
2. `src/dlm/model.py` now explicitly moves tensor batch fields used in
   `training_step` to `self.device`. Without this, formula conditioning could
   construct CPU condition tensors while module weights were on CUDA.

## Failed Launch Attempts

Earlier run directories are retained as diagnostic evidence:

- `dlm_mist_fingerprint_adaptation_20260630T050223Z`
  - failed because the config used a 768-dimensional DLM while `DLM.ckpt` is
    896-dimensional;
  - after architecture overrides, exposed the CPU/CUDA batch-device issue.
- `dlm_mist_fingerprint_adaptation_20260630T050643Z`
  - used the corrected architecture and device fix;
  - failed with CUDA OOM at batch size 512 on A100 80GB.

## Canonical Training Run

Current command is captured in:

```text
runs/dlm_mist_fingerprint_adaptation_20260630T050906Z/run_dlm_mist_adaptation.sh
```

Important settings:

- `configs/fp2mol_finetune_mist_fingerprints.yaml`
- `load_weights_only=/home/nikolenko/work/Projects/FRIGID/repro_cache/DLM.ckpt`
- `data.predicted_fingerprint_key=mist_binary`
- `loader.global_batch_size=256`
- `trainer.devices=1`
- `trainer.num_nodes=1`
- `WANDB_MODE=offline`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

The run started successfully in tmux and reached the training loop. Latest live
check:

```text
Epoch 0: about 81/747 batches
train_total_loss ~= 2.5-2.8
GPU utilization: 94-100%
GPU memory: about 66 GB / 80 GB
checkpoint count: 0
```

The first checkpoint is expected at step 2,500 according to the config. At the
current speed, that checkpoint is not immediate; evaluation should wait until a
checkpoint exists because running DLM robustness evaluation on the same A100
while training is active would likely compete for memory.

## Monitoring

Use:

```bash
ssh spectrum
tmux attach -t frigid_dlm_mist_adapt
```

or inspect logs directly:

```bash
tail -f /home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dlm_mist_fingerprint_adaptation_20260630T050906Z/train.log
```

Checkpoints will be under:

```text
/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dlm_mist_fingerprint_adaptation_20260630T050906Z/train/checkpoints
```

## Next Evaluation

After at least one checkpoint is produced, evaluate an adapted checkpoint with
the paired robustness benchmark. Do not launch this concurrently with the active
training process on the same GPU.

```bash
python scripts/benchmark_dlm_fingerprint_robustness.py \
  --config configs/spec2mol_benchmark_msg.yaml \
  --data-dir /home/nikolenko/work/Projects/FRIGID/repro_cache/msg \
  --mist-checkpoint /home/nikolenko/work/Projects/FRIGID/repro_cache/mist_msg.pt \
  --dlm-checkpoint <adapted_checkpoint> \
  --use-shared-cross-attention \
  --formula-matches 10 \
  --max-attempts 100 \
  --batch-size 16 \
  --partial-save-every 100 \
  --fingerprint-sources ground_truth mist_binary \
  --max-spectra 1400 \
  --output-dir runs/dlm_mist_adapted_robustness_1400
```

The decision metric is whether the adapted DLM improves `mist_binary` decoding
quality relative to the original DLM without collapsing `ground_truth`
conditioning.
