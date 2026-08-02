# MARLIN autoresearch v2

## Objective

Obtain reproducible non-zero Exact@1 and Exact@10 on structure-disjoint NPLIB1
validation using spectrum-derived DreaMS fingerprints and precursor mass. The
locked 803-spectrum test remains unused until the recipe passes validation
with at least three seeds.

## Mechanistic baseline

The step-30000 decoder is not globally incapable of molecular retrieval. The
saved oracle diagnostic
`/mnt/netstorage/nikolenko/marlin/runs/sampler-diagnosis/step30000-oracle-raw`
used 32 spectra, 16 candidates, raw weights, grammar masking and mass-shell
decoding and produced:

- Exact@1: 0.03125;
- Exact@10: 0.03125;
- validity: 0.572265625;
- mass validity: 0.027318948412698413;
- candidate return rate: 0.1875.

This is a diagnostic upper bound only and cannot be promoted. Honest
spectrum-derived DreaMS runs remained at zero Exact and zero mass-shell return,
so the current falsifiable mechanism is conditioning-distribution mismatch
between training fingerprints and predicted DreaMS fingerprints.

## Fixed search contract

- early panel: `nplib1_val_micro32_v1.tsv`;
- confirmation panels: nested micro64 and molecule-disjoint macro64;
- fingerprint input: DreaMS `probs`, threshold 0.95;
- screening decoding: 16 candidates, seed 42, grammar and mass shell on;
- paper-comparable confirmation: 384 candidates per spectrum, matching the
  paper, on at least three fixed seeds;
- final metrics: Exact@1 and Exact@10;
- search metric while Exact is zero: lexicographic validity, mass-valid
  decoding, strict candidate return, then Exact.

The 16-candidate lane is only a technical/futility screen. It cannot establish
an incumbent or be compared with the paper's 384-decode Top-1/Top-10 results.
Promotion requires the unchanged paper-comparable inference settings:
block width 8, 10 ppm mass acceptance, and conditioning-diversity dropout 0.3.

Ground-truth and oracle fingerprints are permitted only in explicitly labelled
diagnostics. They cannot establish an incumbent.

## Active serialized run

Slurm `609` is retained only as an infrastructure failure: it reached the
training launcher while `/mnt/netstorage` was full and failed before writing
`run_manifest.json`. Its absence of molecular metrics must not be interpreted
as a model result. The dependent stale jobs `610` and `611` were cancelled.

The storage-only replacement `616` was stopped after a partial `4/32` rows:
the 32-row evaluator would not fit the two-hour wall clock. It is retained as
an infrastructure control and has no aggregate molecular score.

The active paper-recipe screen is Slurm `617`, using the same checkpoint,
evaluator and input hashes, routing outputs and offline ClearML cache to local
disk, and enabling symmetric fingerprint noise (`p=0.5`, `rho~U(0.1,0.3)`).
It runs on a free `gpu-shared` shard and does not preempt the foreign job on
the shared A100. The screen is bounded to four spectra and 16 candidates so
that a complete evaluator artifact fits the two-hour wall clock.

- log:
  `/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/marlin-fp-adapt-617.out`;
- run root:
  `/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/runs/spectrum-fingerprint-adaptation-slurm-617`;
- expected molecular artifact:
  `/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/runs/spectrum-fingerprint-adaptation-slurm-617/periodic_molecular/step=100/metrics.json`;
- offline ClearML task:
  `offline-bf2f93df2ff741caa4cf4b155624ba3c` (the 617 task id is recorded in
  its run manifest after ClearML initialization).

The numbered audit of this run and its predecessor is maintained in
`docs/MARLIN_EXPERIMENT_REPORT_RU.md`.

## Decision after job 611

1. If validity remains zero, inspect termination, grammar dead ends and
   teacher-forced reconstruction before further conditioning experiments.
2. If validity is non-zero but mass validity is zero, optimize predicted-
   fingerprint conditioning and mass-compatible termination; do not tune
   ranking or Exact sampling yet.
3. If strict candidate return is non-zero, compare Tanimoto/formula recall and
   train the smallest attributable conditioning change.
4. If Exact becomes non-zero, repeat unchanged on three seeds, then micro64 and
   macro64. Only a stable candidate advances to full validation-396.

The trusted autoresearch controller currently exposes only a CPU-isolated
`bubblewrap` evaluator. It cannot serve as promotion authority for A100 model
evaluation until a verified GPU runtime adapter exists. Slurm receipts,
immutable scorer commits, run signatures, dataset hashes and artifacts are the
interim audit trail; no CPU surrogate may replace molecular generation.
