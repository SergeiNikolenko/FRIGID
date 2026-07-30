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
- decoding: 16 candidates, seed 42 for screening, grammar and mass shell on;
- confirmation: at least three fixed seeds;
- final metrics: Exact@1 and Exact@10;
- search metric while Exact is zero: lexicographic validity, mass-valid
  decoding, strict candidate return, then Exact.

Ground-truth and oracle fingerprints are permitted only in explicitly labelled
diagnostics. They cannot establish an incumbent.

## Active serialized run

- Slurm 609: 100-step cross-attention adaptation to predicted DreaMS
  fingerprints, queued after the shared GPU predecessor.
- Slurm 611: scorer revision `4926f43d3c93320f9469ebc5547d61ed7cfeef0d`,
  `afterok:609`, fixed micro32/16-candidate/seed-42 evaluation.
- Expected metric artifact:
  `/mnt/netstorage/nikolenko/marlin/runs/autoresearch/v2/job609-micro32-seed42.json`.

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
