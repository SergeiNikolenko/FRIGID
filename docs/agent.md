# FRIGID project evidence

Start with [the archive guide](FRIGID_project/README.md), the
[technical audit](FRIGID_TECHNICAL_AUDIT.md), and
[benchmark system](FRIGID_BENCHMARK_SYSTEM.md).

The source tree keeps compact reports and baseline summaries. Large historical
run artifacts are preserved separately on the existing Gauss research host,
with original and portable archives and verified SHA-256 manifests.

The portable archive preserves every original regular file. Twelve internal
links resolve relatively; 252 external dataset references are explicitly
listed as unavailable. It is an evidence package, not a complete runnable
training dataset. Do not claim that archive portability repairs the historical
benchmark's missing rows, denominator errors, or scientific limitations.

Use [operational usage](FRIGID_OPERATIONAL_USAGE.md) for stable CLI entry points.
Keep inspection and lightweight smoke checks local; run training and heavy
benchmarks on an authorized compute host. Preserve original archives and Git
history when repackaging results.

## Experiment documentation

The following experiment statuses are June 2026 records, not a live scheduler
view. Check current jobs and checkpoints before continuing a run. The opt-in
DreaMS workflows preserve the existing MIST/DLM production entry points.

## Operational Docs

- `docs/FRIGID_EXPERIMENT_REGISTRY.md`: top-level experiment index with status,
  plan/result links, decisions, and the active Linear project URL.
- `docs/FRIGID_EXPERIMENT_STATUS_AND_IDEAS.md`: historical status board for June 2026 runs,
  near-term queue, backlog ideas, blockers, stop rules, and Linear issue IDs.
- `docs/FRIGID_EXPERIMENT_TEMPLATE.md`: required template for new experiment
  plans, run reports, and artifact manifests.
- `docs/FRIGID_OPERATIONAL_USAGE.md`: CLI entry points, expected inputs,
  benchmark outputs, DLM robustness workflow, and MIST-fingerprint DLM
  adaptation commands.
- `docs/DLM_FINGERPRINT_ROBUSTNESS_RESULTS.md`: paired MIST-to-DLM diagnostic
  results. This includes the 64-spectrum diagnostic and the 1,400-spectrum
  partial run showing a large quality drop when DLM receives MIST-predicted
  fingerprints instead of ground-truth Morgan fingerprints.
- `docs/FRIGID_DREAMS_FINGERPRINT_HEAD.md`: overall DreaMS fingerprint-head
  workflow and decision gates.
- `docs/FRIGID_DREAMS_ADAPTER_REPLACEMENT_PLAN_20260630.md`: planned next
  DreaMS replacement path using staged adapter adaptation, MIST anchoring,
  MIST-error focus, and decoder-sensitive weighting.
- `docs/FRIGID_dreams_adapter_replacement_20260630/`: runnable DreaMS adapter
  replacement experiment package with the plan, Slurm launcher, run report,
  and artifact manifest.
- `docs/FRIGID_spectral_jepa_pretraining_20260630/`: runnable JEPA-style
  self-supervised spectrum pretraining package with plan, Slurm launcher, run
  report, and artifact manifest. This is a representation-learning experiment,
  not a direct MIST replacement until it passes the fingerprint gate.

## DreaMS Encoder Experiments

- `docs/FRIGID_dreams_fingerprint_head_20260629/`: full MSG train/validation
  frozen-DreaMS embedding experiment. Result: frozen DreaMS embeddings plus a
  shallow fingerprint head were not competitive with MIST.
- `docs/FRIGID_dreams_fingerprint_head_round2_20260629/`: second-round
  calibration/loss matrix for frozen DreaMS fingerprint heads.

## MIST + DreaMS Combination Experiments

- `docs/FRIGID_mist_dreams_blend_20260629/`: linear probability blend between
  MIST and the best DreaMS head. Result: best alpha was pure MIST.
- `docs/FRIGID_mist_dreams_residual_20260629/`: learned residual adapter using
  DreaMS embeddings and MIST logits. Result: unanchored residuals drifted below
  MIST; anchored residuals produced a small positive delta but did not clear the
  encoder gate.
- `docs/FRIGID_dreams_distilled_replacement_20260629/`: DreaMS-only replacement
  candidate trained with MIST distillation and MIST-error-focused supervision.
  Result: best variant remained roughly 0.30 mean Tanimoto below MIST.
- `docs/FRIGID_dreams_full_finetune_20260629/`: full DreaMS encoder plus
  fingerprint-head fine-tune on the complete FRIGID/MassSpecGym train split.
- `docs/FRIGID_DREAMS_ADAPTER_REPLACEMENT_PLAN_20260630.md`: next proposed
  DreaMS replacement experiment. This should be treated as the serious DreaMS
  continuation, not another repeat of the failed frozen-head or plain
  full-fine-tune runs.
- `docs/FRIGID_dreams_adapter_replacement_20260630/`: historical package for the
  serious DreaMS continuation. Use this directory for launch status and
  artifacts.
- `docs/FRIGID_NEXT_EXPERIMENT_DECISION_20260630.md`: decision memo selecting
  DLM adaptation to MIST-predicted fingerprints as the next serious experiment.
- `docs/FRIGID_dlm_mist_fingerprint_adaptation_20260630/`: historical DLM
  adaptation run on full train-split MIST binary fingerprints, including launch
  fixes, run directory, monitoring, checkpoint schedule, and next evaluation
  command.
- `docs/FRIGID_spectral_jepa_pretraining_20260630/`: planned JEPA-like spectrum
  representation experiment. It should run after active GPU jobs clear and be
  judged first by frozen-encoder fingerprint metrics.
