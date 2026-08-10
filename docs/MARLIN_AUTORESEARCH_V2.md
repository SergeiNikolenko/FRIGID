# MARLIN autoresearch v2

## Objective

Obtain reproducible non-zero Exact@1 and Exact@10 on structure-disjoint NPLIB1
validation using spectrum-derived DreaMS fingerprints and precursor mass.

The locked 803-spectrum test split is reserved for final confirmation only. It is
not a search surface: no recipe, hyperparameter, threshold, panel or checkpoint
may be chosen against it, and every intermediate decision belongs on the
396-spectrum validation panel.

**It has already been consumed once, and this is recorded rather than hidden.**
The step=100000 checkpoint of the FRIGID warm-start run was evaluated on the full
locked split at 8 candidates, seed 42, with the mass prune and both vocabulary
restrictions, returning `Exact@1 = 2.74%` and `Exact@10 = 3.24%`
(`docs/MARLIN_STATUS_REPORT.md`). That measurement stands as the reproduction's
headline number. Because the checkpoint that produced it was itself picked
without a held-out selection signal, the split has paid for one arbitrary draw;
any further use must be a pre-declared confirmation of a recipe already settled
on validation, and must be added to the ledger below.

### Locked test split usage ledger

| date | checkpoint | candidates | seed | result | authorization |
| --- | --- | ---: | ---: | --- | --- |
| 10 August 2026 | FRIGID warm start, step=100000 | 8 | 42 | `Exact@1 2.74%`, `Exact@10 3.24%` | first and so far only use |

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

- search panel: `nplib1_val_full396_v1.tsv`, the whole validation split, sharded
  across cores;
- confirmation: the same 396 spectra at a higher candidate budget, on at least
  three fixed seeds;
- final confirmation: the locked 803-spectrum test split, once, on a recipe that
  is already settled;
- fingerprint input: DreaMS `probs`, threshold 0.95;
- screening decoding: 8 candidates, seed 42, grammar mask, mass shell, mass
  reachability prune, isotope-token ban and CHNOPS-plus-halogen restriction;
- paper-comparable confirmation: 384 candidates per spectrum, matching the paper;
- final metrics: Exact@1 and Exact@10;
- selection metric during a run: candidate return rate. Never Exact@k.

The 32-spectrum micro panel is retired and must not be used for any decision. Its
floor is one molecule in 32, it swings by up to 0.12 between neighbouring
checkpoints, and several claims in `docs/MARLIN_STATUS_REPORT.md` were published
and then withdrawn because that noise was read as signal. It existed only because
a full split looked unaffordable, which is no longer true: the constrained decoder
is CPU bound, so `scripts/evaluate_marlin_sharded.sh` and the periodic evaluation
split a panel across cores at close to linear speedup. The 803-spectrum split at 8
candidates takes about 1.1 h on 16 shards against about 17 h in one process, which
puts the 396-spectrum validation panel at roughly 30 min per checkpoint.

Screening at 8 candidates replaces the earlier 16-candidate lane: 8 is the budget
every sharded cost figure and every matched re-evaluation in the status report was
measured at, so keeping it makes those numbers comparable. It remains a technical
screen. It cannot establish an incumbent or be compared with the paper's
384-decode Top-1/Top-10 results. Promotion requires the unchanged
paper-comparable inference settings: block width 8, 10 ppm mass acceptance, and
conditioning-diversity dropout 0.3.

## Held-out signal during a run

Before 11 August 2026 an adaptation run had no held-out signal it could act on: no
validation loss, no early stopping, and a periodic panel of 32 spectra whose
resolution floor sat above the effect being looked for. The 100,000-step
warm-start run therefore reported a flat `Exact@1 = 0.0312` from step 30,000
onward while a matched offline re-evaluation of the same checkpoints showed it
degrading. Three metrics moved together:

| matched re-evaluation | 20k | 30k | 40k | 50k | 60k | 70k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| candidate return | **0.3438** | 0.2812 | 0.2188 | 0.1562 | 0.3125 | 0.1875 |
| uniqueness | **0.3333** | 0.2634 | 0.1979 | 0.1354 | 0.2917 | 0.1523 |
| mass validity | **0.2419** | 0.1594 | 0.1161 | 0.0987 | 0.1990 | 0.1542 |
| Exact@1 | 0.0312 | 0.0312 | 0.0312 | 0.0312 | 0.0312 | 0.0312 |

The contract is now:

- **candidate return rate** drives checkpoint selection and early stopping. It
  halved from its step-20,000 peak while `Exact@1` did not move at all, its
  denominator is every spectrum on the panel, and it is the coverage failure the
  status report identifies as the binding one. Uniqueness and mass validity are
  admitted alternatives; `Exact@1` and `Exact@10` are refused in code, because at
  396 spectra their resolution is still 0.25% and on 32 spectra they were flat
  while the model degraded;
- patience is configurable and defaults to 3 evaluations. On the trajectory above
  a patience of 3 stops after step 50,000 and keeps step 20,000, which is where
  the status report places the useful checkpoint;
- a **validation loss** is logged at every evaluation interval: the same
  masked-diffusion objective, on a structure-disjoint 5% slice of the adaptation
  set carved by InChIKey connectivity block. It is taken from the adaptation set
  rather than from the validation split because the validation split *is* the
  396-spectrum molecular panel, and a loss measured there would consume the panel
  it is meant to complement. The fingerprint is not corrupted and the mask
  generator is re-seeded every pass, so consecutive checkpoints differ only by
  their weights;
- evaluation stays fail-soft. A failed shard or an unusable metric is recorded in
  `periodic_molecular/step=<n>/failure.json` and never stops training, and a
  partial panel is refused rather than scored, because it is not comparable with a
  full one.

The knobs are `--evaluation-shards`, `--validation-loss-fraction`,
`--select-best-checkpoint`, `--selection-metric`, `--selection-patience` and
`--selection-min-delta` on `scripts/train_marlin_spectrum_adaptation.py`. The best
checkpoint and the full selection history are written to
`<run>/selection/{best.ckpt,selection.json}`.

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
The current training factor is Slurm `635`: train and evaluate with soft DreaMS
amplitudes on the structure-disjoint train/held-out split (offline ClearML task
`offline-d16a7c6cbba1427c9948d35f32bba997`).

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
4. If Exact becomes non-zero, repeat unchanged on three seeds on the
   396-spectrum validation panel, then raise the candidate budget on the same
   panel. The nested micro64 and macro64 escalation is retired with the micro
   panel: a run now starts on the full validation split, so there is no smaller
   panel to be promoted from. Only a recipe stable across those seeds may be
   confirmed once on the locked 803-spectrum test split.

The trusted autoresearch controller currently exposes only a CPU-isolated
`bubblewrap` evaluator. It cannot serve as promotion authority for A100 model
evaluation until a verified GPU runtime adapter exists. Slurm receipts,
immutable scorer commits, run signatures, dataset hashes and artifacts are the
interim audit trail; no CPU surrogate may replace molecular generation.
