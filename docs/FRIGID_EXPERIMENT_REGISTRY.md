# FRIGID Experiment Registry

Date: 2026-06-30

This is the top-level index for FRIGID experiment records. It should answer:

- What was tried?
- Where is the initial plan?
- Where are the results?
- What decision did the experiment support?
- What is the current status?

## Status Legend

- `planned`: designed but not launched.
- `running`: training, evaluation, or transfer is currently active.
- `complete`: run finished and results are recorded.
- `stopped`: run was intentionally stopped because an upstream gate failed.
- `blocked`: work cannot continue without data, infrastructure, or a decision.
- `superseded`: useful historical evidence, but no longer the active path.

## Current Priority

The current serious path is DLM adaptation to MIST-predicted fingerprints. Direct
DreaMS replacement is not the active path because multiple DreaMS-side
experiments failed to approach the MIST fingerprint baseline.

## Experiment Index

| ID | Status | Question | Plan | Results | Decision |
| --- | --- | --- | --- | --- | --- |
| `dlm_mist_fingerprint_adaptation_20260630` | running | Can DLM be adapted to decode the fingerprint distribution that MIST actually emits? | [run report](FRIGID_dlm_mist_fingerprint_adaptation_20260630/run_report.md) | first checkpoint exists, evaluation pending | Active path |
| `dlm_fingerprint_robustness` | complete | How much quality does DLM lose when clean Morgan fingerprints are replaced by MIST-predicted fingerprints? | [operational usage](FRIGID_OPERATIONAL_USAGE.md) | [results](DLM_FINGERPRINT_ROBUSTNESS_RESULTS.md) | Motivated DLM adaptation |
| `dreams_fingerprint_head_20260629` | stopped | Can frozen DreaMS embeddings plus an MLP predict Morgan fingerprints well enough to replace MIST? | [workflow](FRIGID_DREAMS_FINGERPRINT_HEAD.md) | [run report](FRIGID_dreams_fingerprint_head_20260629/run_report.md) | Failed upstream fingerprint gate |
| `dreams_fingerprint_head_round2_20260629` | complete | Do calibration and loss variants fix the frozen-DreaMS fingerprint head? | [plan](FRIGID_dreams_fingerprint_head_round2_20260629/experiment_plan.md) | see experiment directory | Did not clear replacement gate |
| `mist_dreams_blend_20260629` | complete | Does a linear probability blend of MIST and DreaMS improve MIST? | [plan](FRIGID_mist_dreams_blend_20260629/experiment_plan.md) | [run report](FRIGID_mist_dreams_blend_20260629/run_report.md) | Best blend was pure MIST |
| `mist_dreams_residual_20260629` | complete | Can DreaMS embeddings correct MIST errors through a residual adapter? | [plan](FRIGID_mist_dreams_residual_20260629/experiment_plan.md) | [run report](FRIGID_mist_dreams_residual_20260629/run_report.md) | Anchored residual gain was too small |
| `dreams_distilled_replacement_20260629` | complete | Can DreaMS replace MIST through distillation and error-focused supervision? | [plan](FRIGID_dreams_distilled_replacement_20260629/experiment_plan.md) | [run report](FRIGID_dreams_distilled_replacement_20260629/run_report.md) | Still far below MIST |
| `dreams_full_finetune_20260629` | complete | Does full DreaMS encoder fine-tuning close the gap? | see run directory | [run report](FRIGID_dreams_full_finetune_20260629/run_report.md) | Improved over frozen heads, still far below MIST |

## Required Record For New Experiments

Every new experiment should have a directory under `docs/` named:

```text
docs/FRIGID_<short_experiment_slug>_<YYYYMMDD>/
```

Each directory should contain:

- `experiment_plan.md`: written before launch. It defines the hypothesis,
  inputs, command, expected runtime, metrics, success criteria, and stop rules.
- `run_report.md`: written during and after the run. It records host, branch,
  commit, run directory, commands, checkpoints, metrics, failures, and decision.
- `artifact_manifest.json`: optional, but recommended for runs with many files.
  It should list copied artifacts, source paths, sizes, and hashes when
  practical.

If an experiment is still active, the current live state belongs in
[FRIGID_EXPERIMENT_STATUS_AND_IDEAS.md](FRIGID_EXPERIMENT_STATUS_AND_IDEAS.md),
not only in chat.

## Decision Rule

Do not promote a model or pipeline change based only on training loss. Promotion
requires a recorded benchmark result against the relevant baseline, with the
same split, fingerprint threshold, generation settings, and sample cap unless
the report explicitly explains why the comparison changed.
