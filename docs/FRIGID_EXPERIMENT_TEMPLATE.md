# FRIGID Experiment Template

Use this template for every new FRIGID experiment directory under `docs/`.
File contents should stay concise but complete enough for another agent to
continue the run without reading chat history.

## experiment_plan.md

````markdown
# <Experiment Name>

Date: <YYYY-MM-DD>

## Question

What concrete hypothesis is this experiment testing?

## Motivation

Why is this the next useful experiment? Link prior evidence.

## Inputs

- Dataset:
- Split:
- Model/checkpoint:
- Baseline:
- Remote host:
- Expected run directory:

## Method

What will be trained or evaluated? Include enough detail to distinguish this
from adjacent experiments.

## Commands

```bash
<exact launch commands>
```

## Metrics

- Primary metric:
- Secondary metrics:
- Required comparison baseline:

## Success Criteria

What result is good enough to continue or promote?

## Stop Rules

What result is bad enough to stop or redirect?

## Expected Artifacts

- Checkpoints:
- Logs:
- Metrics JSON/CSV:
- Reports:
````

## run_report.md

````markdown
# <Experiment Name> Run Report

Date: <YYYY-MM-DD>

## Summary

One paragraph with the final result or current live status.

## Run Identity

- Host:
- Branch:
- Commit:
- Run directory:
- tmux/session:
- Command file:

## Inputs

- Dataset:
- Split:
- Checkpoints:
- Config:

## What Happened

Record launch attempts, failures, fixes, restarts, and important observations.

## Metrics

| Metric | Baseline | This Run | Delta |
| --- | ---: | ---: | ---: |
| <metric> | | | |

## Artifacts

- Logs:
- Checkpoints:
- Metrics:
- Partial outputs:

## Interpretation

What does the result mean for FRIGID?

## Decision

Continue, stop, repeat with changes, or promote.

## Next Actions

1. <next action>
````

## artifact_manifest.json

```json
{
  "experiment_id": "<slug_YYYYMMDD>",
  "source_host": "<host>",
  "source_run_dir": "<absolute path>",
  "artifacts": [
    {
      "name": "<artifact name>",
      "path": "<absolute or docs-relative path>",
      "type": "checkpoint|metrics|log|report|dataset|script",
      "size_bytes": null,
      "sha256": null,
      "notes": ""
    }
  ]
}
```
