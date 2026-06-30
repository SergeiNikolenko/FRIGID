# FRIGID Agent Documentation Index

This file is the agent-facing index for FRIGID project documentation and
experiment artifacts.

## Operational Docs

- `docs/FRIGID_EXPERIMENT_REGISTRY.md`: top-level experiment index with status,
  plan/result links, decisions, and the active Linear project URL.
- `docs/FRIGID_EXPERIMENT_STATUS_AND_IDEAS.md`: live board for active runs,
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
- `docs/FRIGID_dreams_adapter_replacement_20260630/`: active package for the
  serious DreaMS continuation. Use this directory for launch status and
  artifacts.
- `docs/FRIGID_NEXT_EXPERIMENT_DECISION_20260630.md`: decision memo selecting
  DLM adaptation to MIST-predicted fingerprints as the next serious experiment.
- `docs/FRIGID_dlm_mist_fingerprint_adaptation_20260630/`: active DLM
  adaptation run on full train-split MIST binary fingerprints, including launch
  fixes, run directory, monitoring, checkpoint schedule, and next evaluation
  command.

## Consolidated Artifact Package

- `docs/FRIGID_project/`: consolidated local artifact package copied from the
  Desktop project bundle. Treat this as a historical artifact handoff package;
  do not duplicate or reorganize it unless explicitly requested.

## Current Decision State

MIST remains the strongest validated spectrum-to-fingerprint path. Direct
DreaMS replacement attempts have failed across frozen heads, distillation,
blend/residual adapters, and plain full encoder fine-tuning. The active run is
DLM adaptation to the MIST-predicted fingerprint distribution, because the
largest measured gap is the DLM quality drop from clean ground-truth
fingerprints to realistic MIST binary fingerprints. The next DreaMS path should
use the adapter replacement plan with MIST anchoring and decoder-sensitive
weighting rather than repeating the previous DreaMS objectives.
