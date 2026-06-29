# FRIGID Agent Documentation Index

This file is the agent-facing index for FRIGID project documentation and
experiment artifacts.

## Operational Docs

- `docs/FRIGID_OPERATIONAL_USAGE.md`: CLI entry points, expected inputs,
  benchmark outputs, DLM robustness workflow, and MIST-fingerprint DLM
  adaptation commands.
- `docs/DLM_FINGERPRINT_ROBUSTNESS_RESULTS.md`: paired MIST-to-DLM diagnostic
  results. This includes the 64-spectrum diagnostic and the 1,400-spectrum
  partial run showing a large quality drop when DLM receives MIST-predicted
  fingerprints instead of ground-truth Morgan fingerprints.
- `docs/FRIGID_DREAMS_FINGERPRINT_HEAD.md`: overall DreaMS fingerprint-head
  workflow and decision gates.

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
  This is the next long-running experiment for testing whether DreaMS can become
  a first-class replacement rather than only a residual helper.

## Consolidated Artifact Package

- `docs/FRIGID_project/`: consolidated local artifact package copied from the
  Desktop project bundle. Treat this as a historical artifact handoff package;
  do not duplicate or reorganize it unless explicitly requested.

## Current Decision State

MIST remains the strongest validated spectrum-to-fingerprint path. DreaMS is
still useful as a research direction, but current evidence does not support a
direct replacement yet. The next serious DreaMS-centered direction is
error-focused or partially fine-tuned training that targets MIST failure regions
instead of another plain frozen-embedding head.
