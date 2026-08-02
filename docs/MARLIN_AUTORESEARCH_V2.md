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

The molecular evaluator must use multinomial token selection for the paper
lane (`--sample-tokens`), with fixed seed and temperature. Argmax is retained
only as an explicit diagnostic and cannot be compared with the stochastic
paper lane.

## Active serialized run

Slurm `609` is retained only as an infrastructure failure: it reached the
training launcher while `/mnt/netstorage` was full and failed before writing
`run_manifest.json`. Its absence of molecular metrics must not be interpreted
as a model result. The dependent stale jobs `610` and `611` were cancelled.

The storage-only replacement `616` was stopped after a partial `4/32` rows:
the 32-row evaluator would not fit the two-hour wall clock. It is retained as
an infrastructure control and has no aggregate molecular score.

Slurm `618` completed the paper-noise adaptation with the corrected micro4
manifest. Its argmax diagnostic produced Exact@1 `0`, Exact@10 `0`, candidate
return `0`, mass validity `0`, strict validity `0.046875`, and validity
`0.046875`; it is rejected as a paper result because the evaluator omitted
multinomial token sampling.

The completed paper-recipe screens are Slurm `619` (multinomial sampling) and
`620` (the same recipe with train threshold aligned to evaluation at `0.95`).
Both used the same checkpoint, evaluator and input hashes, local-disk outputs,
symmetric fingerprint noise (`p=0.5`, `rho~U(0.1,0.3)`), and a four-row held-out
panel. Both returned Exact@1/10 `0` and candidate return `0`; their full
metrics remain in the numbered experiment ledger.

Two inference-only diagnostics are serialized next: `623` disables the mass
shell to test whether it is the immediate bottleneck, while `624` uses the
canvas decoder with the shell enabled. They are not promotion candidates and
must be reported with their exact mode and panel.

The full-backbone causal training test `625` is complete: 100
cross-attention-only steps then 900 full-backbone steps improved grammar
validity to `0.140625`, but Exact@1/10 and mass-compatible return stayed `0`.
The serialized inference follow-ups are `628` (threshold `0.50`, negative),
`629` (soft DreaMS confidence, negative), and `630` (EOS boost, running).
The next queued training factor is `631`: train and evaluate with soft DreaMS
amplitudes on the structure-disjoint train/held-out split.

- logs:
  `/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/logs/marlin-fp-adapt-{619,620,625}.out`;
- run roots:
  `/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/runs/spectrum-fingerprint-adaptation-slurm-{619,620,625}`;
- molecular artifacts:
  `.../periodic_molecular/step=100/metrics.json` for `619/620`, and
  `.../periodic_molecular/step=1000/metrics.json` for `625`;
- inference artifacts for `628` and `629` are under
  `/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/runs/marlin-ablate-{threshold050,soft050}-{628,629}`;
- offline ClearML task IDs are recorded in each run's `run_manifest.json`;
  the API-backed ClearML dashboard is still blocked by the private API auth
  proxy, so offline IDs are not presented as live web tasks.

The numbered audit of this run and its predecessor is maintained in
`docs/MARLIN_EXPERIMENT_REPORT_RU.md`.

## Decision after job 620

1. If the no-shell/canvas diagnostics show valid candidates but no mass return,
   keep the shell in the paper lane and fix mass-compatible termination rather
   than claiming an Exact improvement from an invalid decoder.
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
