# FRIGID DreaMS Adapter Replacement Plan

Date: 2026-06-30

## Objective

Design the next serious DreaMS experiment for FRIGID after the first direct
DreaMS replacement attempts failed. The goal is not to repeat another frozen
embedding head or plain full fine-tune. The goal is to test whether DreaMS can
be adapted into a decoder-compatible spectrum-to-fingerprint replacement for
the current MIST backend.

Target interface:

```text
MS/MS spectrum -> DreaMS adapter encoder -> decoder-compatible fingerprint -> DLM
```

## Evidence Already Available

The current strongest validated encoder baseline is MIST on the MSG validation
split:

```text
mean fingerprint Tanimoto ~= 0.5420
```

Existing DreaMS-side experiments:

| Experiment | Best result | Decision |
| --- | ---: | --- |
| Frozen DreaMS embedding head | 0.1238 mean validation Tanimoto | Failed replacement gate |
| Frozen DreaMS head with loss/calibration variants | about 0.234 best pure DreaMS point | Improved but still far below MIST |
| DreaMS-only MIST-distilled head | 0.2404 mean validation Tanimoto | Failed replacement gate |
| Full DreaMS encoder fine-tune | 0.2580 validation Tanimoto, strong overfit | Failed replacement gate |
| Linear MIST + DreaMS blend | best alpha was pure MIST | No DreaMS gain |
| MIST + DreaMS anchored residual | +0.00068 over MIST | Real but too small |

These results do not prove that DreaMS is unusable. They show that the previous
objectives were not sufficient:

- frozen DreaMS embeddings do not carry enough linearly accessible Morgan
  fingerprint signal;
- plain full unfreezing overfits and still remains far below MIST;
- residual correction can find a direction above MIST, but the effect is too
  small under the current objective;
- optimizing plain Morgan fingerprint Tanimoto may not be aligned enough with
  DLM molecule-generation quality.

## New Hypothesis

DreaMS may help if it is adapted with a staged objective that keeps the useful
MIST-like structure while focusing DreaMS capacity on decoder-relevant MIST
errors.

The experiment should therefore use:

1. staged adapter training instead of immediately unfreezing the whole DreaMS
   backbone;
2. a MIST anchor to avoid drifting below the strong MIST solution;
3. explicit MIST-error and uncertainty masks;
4. sparsity control because earlier DreaMS heads overpredicted active bits;
5. decoder-sensitive weights derived from the DLM robustness gap.

## Proposed Experiment

Experiment ID:

```text
dreams_adapter_replacement_20260630
```

Remote host:

```text
spectrum
```

Execution mode:

```text
Slurm, partition gpu, GRES gpu:a100:1
```

Data:

- MassSpecGym train split: 191,216 spectra.
- MassSpecGym validation split: 19,043 spectra.
- Same MSG split, metadata, Morgan fingerprint target, MIST checkpoint, and
  thresholding conventions as the existing DreaMS and MIST experiments.

Model:

```text
pretrained DreaMS encoder
  -> small trainable adapter / upper-layer adapter
  -> 4096-bit fingerprint head
```

Training stages:

1. Warm up only the adapter and fingerprint head.
2. Unfreeze only the top DreaMS blocks or adapter-inserted layers.
3. Keep the encoder learning rate much lower than the head learning rate.
4. Select checkpoints by full-validation metrics, not training loss.

Initial learning-rate shape:

```text
head_lr: 1e-3
adapter_lr: 3e-4
encoder_top_lr: 1e-5 to 3e-5
```

## Loss Terms

Use a weighted objective:

```text
loss =
  target_bce
  + soft_tanimoto
  + active_count_penalty
  + mist_distillation_anchor
  + mist_error_focus
  + decoder_sensitive_weighting
```

Terms:

- `target_bce`: supervised Morgan fingerprint target.
- `soft_tanimoto`: direct overlap objective.
- `active_count_penalty`: penalizes predicted active-bit count drift.
- `mist_distillation_anchor`: keeps predictions close to MIST where MIST is
  already strong.
- `mist_error_focus`: upweights bits where MIST is wrong or uncertain.
- `decoder_sensitive_weighting`: upweights spectra or bits that caused the
  largest DLM degradation in paired robustness runs.

This differs from the previous distilled replacement run because MIST is not
only a teacher. It is an anchor plus an error map. The model should learn where
to preserve MIST and where to depart from it.

## Evaluation Gates

### Encoder Gate

Run on the full validation split.

Promotion threshold:

```text
mean validation fingerprint Tanimoto >= MIST + 0.005
```

With the current MIST baseline, this means approximately:

```text
mean validation fingerprint Tanimoto >= 0.547
```

Secondary checks:

- median Tanimoto;
- bit precision, recall, and F1;
- mean predicted active bits;
- improvement on MIST-error cases;
- no large regression on MIST-easy cases.

### DLM Gate

Run only if the encoder gate passes, or if the encoder result is close enough
to justify a diagnostic run.

Use paired DLM robustness evaluation with:

```text
fingerprint sources:
  ground_truth
  mist_binary
  dreams_adapter_binary
```

Decision metric:

- `dreams_adapter_binary` must improve DLM Exact@1 and Tanimoto@1 versus
  `mist_binary` under the same DLM checkpoint and generation settings.
- It must not only improve fingerprint Tanimoto while damaging molecular
  generation.

## Stop Rules

Stop this DreaMS replacement path if:

- full validation fingerprint Tanimoto remains below 0.45 after staged adapter
  fine-tuning;
- validation overfit repeats the previous full fine-tune pattern;
- the model only improves MIST-error cases by harming MIST-easy cases enough to
  lose global quality;
- DLM paired evaluation does not improve over `mist_binary`.

## Required Artifacts

Before launch:

- `docs/FRIGID_dreams_adapter_replacement_20260630/experiment_plan.md`
- Slurm submission script copied into that directory.

During and after run:

- ClearML task URL, if online tracking is used.
- Slurm job id.
- exact branch and commit.
- remote run directory.
- training config.
- validation summary JSON.
- calibration sweep CSV/JSON.
- checkpoint manifest.
- final `run_report.md`.

## Relationship To The Current DLM Adaptation Run

The current DLM adaptation to MIST fingerprints remains valuable. It tests
whether the decoder can adapt to the existing MIST output distribution. The
DreaMS adapter replacement plan tests a different question: whether the encoder
side can produce a better decoder-compatible distribution than MIST.

These two paths should be compared downstream using the same paired DLM
robustness benchmark.
