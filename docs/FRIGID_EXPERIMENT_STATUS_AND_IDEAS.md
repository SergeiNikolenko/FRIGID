# FRIGID Experiment Status And Ideas

Date: 2026-06-30

This document is the live experiment board. It should be updated whenever a run
starts, fails, produces a checkpoint, finishes evaluation, or changes the next
decision.

## Linear Tracking

- Linear project: https://linear.app/nikolenko-sergei-s/project/frigid-d174fd1451a6
- Active monitoring issue: `SPA-74`.
- Benchmark evaluation issue: `SPA-75`.
- Documentation update issue: `SPA-76`.
- Artifact-transfer issue: `SPA-77`.
- Next-objective design issue: `SPA-78`.
- DreaMS evidence guardrail issue: `SPA-79`.

## Active Runs

### DreaMS Adapter Replacement

- Status: `queued`.
- Host: `spectrum`.
- Slurm job: `34`.
- Dependency: `afterany:32`, so it waits for the active DLM adaptation job to
  leave the A100.
- Partition: `gpu`.
- GRES: `gpu:a100:1`.
- Requested resources: 16 CPU, 120G memory, one A100.
- Remote checkout:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head`
- Branch at submission: `dreams-fingerprint-head`.
- Submit commit: `9f8a55e`.
- Latest verified status: 2026-06-30 14:41 UTC.
- Command file:
  `docs/FRIGID_dreams_adapter_replacement_20260630/run_dreams_adapter_replacement.sbatch`
- Slurm logs:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/slurm_logs/frigid-dreams-adapter-34.out`
  and
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/slurm_logs/frigid-dreams-adapter-34.err`
- Expected run directory:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_adapter_replacement_<UTC>`
- Current state at last check: still `PENDING` with reason `Dependency`;
  no DreaMS run directory or Slurm log has been created yet.
- Next action: after job `32` ends, confirm job `34` starts, writes
  `run_identity.json`, and completes the first validation sweep.

### DLM Adaptation To MIST Fingerprints

- Status: `running`.
- Host: `spectrum`.
- Training tmux: `frigid_dlm_mist_adapt`.
- Monitor tmux: `frigid_dlm_mist_monitor`.
- Slurm job: `32`, running on partition `gpu` with `gres/gpu:a100:1`.
- Superseded Slurm job: `16`, which briefly overlapped the tmux process and
  failed with CUDA OOM.
- Run directory:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dlm_mist_fingerprint_adaptation_20260630T050906Z`
- Branch at launch: `dreams-fingerprint-head`.
- Launch commit: `44a5ff0`.
- Latest documented code commit: `8a7ca8b`.
- Latest verified status: 2026-06-30 14:41 UTC.
- Progress at last check: Slurm job `32` was still running normally at epoch
  19, about 159/747 batches, with `train_total_loss` around 0.222 in the latest
  parsed progress line.
- Latest checkpoint:
  `/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dlm_mist_fingerprint_adaptation_20260630T050906Z/train/checkpoints/12500.ckpt`
- Checkpoints currently present: `2500.ckpt`, `5000.ckpt`, `7500.ckpt`,
  `10000.ckpt`, and `12500.ckpt`.
- Current blocker: training is now under Slurm; benchmark evaluation still
  waits for a selected adapted checkpoint and a free evaluation window.
- Next action: run paired robustness evaluation on an adapted checkpoint without
  `--use-shared-cross-attention`.

## Current Decision State

MIST remains the strongest validated spectrum-to-fingerprint encoder. Direct
DreaMS replacement with the previous frozen-head, distilled-head, blend,
residual, and plain full-fine-tune objectives is paused. The active hypothesis is that the
largest current system loss comes from DLM being trained for clean
ground-truth-like Morgan fingerprints while deployment uses MIST-predicted
binary fingerprints.

A new DreaMS replacement plan exists, but it changes the objective to staged
adapter adaptation with MIST anchoring, MIST-error focus, and decoder-sensitive
weighting:
[FRIGID_dreams_adapter_replacement_20260630/experiment_plan.md](FRIGID_dreams_adapter_replacement_20260630/experiment_plan.md).

The corresponding Slurm job is now queued as job `34`, waiting on DLM job `32`.

## Near-Term Queue

1. Finish DLM adaptation checkpoint evaluation.
   - Compare adapted DLM against original DLM on `ground_truth` and
     `mist_binary`.
   - Start with the same 1,400-spectrum cap used by the prior robustness run.
   - Promote only if `mist_binary` top-1 Tanimoto improves materially without
     collapsing `ground_truth` decoding.
2. If adapted DLM improves:
   - run a larger or full test split robustness benchmark;
   - document generation settings and wall-clock runtime;
   - consider a later checkpoint comparison, for example step 5,000 vs 2,500.
3. If adapted DLM does not improve:
   - treat this as evidence that MIST loses decoder-critical structure before
     DLM sees the fingerprint;
   - move to MIST-side objectives or a decoder-compatible fingerprint target.

## Backlog Ideas

### Decoder-Side Ideas

- Compare DLM checkpoints across training steps, not just the first checkpoint.
- Add validation sampling during DLM adaptation if runtime permits.
- Train on a mixture of `ground_truth` and `mist_binary` fingerprints to avoid
  destroying clean-fingerprint decoding.
- Train with soft `mist_probs` instead of `mist_binary`, then benchmark
  `mist_probs` and `mist_binary` separately.

### Encoder-Side Ideas

- Keep MIST as the baseline and test MIST-side objectives that better match DLM
  decoding quality, not only fingerprint Tanimoto.
- Revisit DreaMS through the planned adapter replacement experiment: staged
  DreaMS adaptation, MIST anchoring, MIST-error focus, and decoder-sensitive
  weighting rather than plain Morgan-bit replacement.
- Test whether DreaMS embeddings help MIST error prediction, but require a gain
  larger than the prior anchored residual result.

### Infrastructure Ideas

- Add a small script that writes a machine-readable experiment status JSON from
  run directories.
- Standardize remote artifact transfer paths, because `spectrum` currently does
  not share `/mnt/ligandpro/shared_storage` with `kolmogorov`.
- Store hashes for large checkpoints before cross-host evaluation.

## Stop Rules

- Stop a DreaMS replacement line when validation fingerprint Tanimoto remains
  far below MIST after a full train/validation run.
- Stop a DLM adaptation line if paired robustness metrics show no meaningful
  improvement on `mist_binary` and a large regression on `ground_truth`.
- Do not keep running expensive DLM generation benchmarks when upstream
  fingerprint metrics already fail the agreed gate.

## Update Protocol

When changing this board:

1. Update `Active Runs` with the real host, session, checkpoint, and run path.
2. Update `Current Decision State` only when benchmark evidence changes the
   conclusion.
3. Move completed experiments into
   [FRIGID_EXPERIMENT_REGISTRY.md](FRIGID_EXPERIMENT_REGISTRY.md).
4. Keep command details in the experiment `run_report.md`, not only here.
