# FRIGID Spectral JEPA Pretraining Experiment

Date: 2026-06-30

## Question

Can a JEPA-style self-supervised spectrum encoder learn a useful MS/MS
representation for FRIGID, then pass the same fingerprint-head gate used for
DreaMS replacement candidates?

This is not a direct replacement for MIST yet. It is a representation-learning
experiment:

```text
MS/MS spectra
  -> JEPA-like pretraining
  -> frozen spectrum encoder
  -> supervised fingerprint head
  -> validation fingerprint gate
```

## Motivation

The direct DreaMS replacement line has not cleared the MIST gate:

- frozen DreaMS head: about `0.1238` mean validation Tanimoto;
- calibrated/loss-tuned frozen DreaMS: about `0.234`;
- DreaMS distilled from MIST: about `0.2404`;
- full DreaMS fine-tune: about `0.2580`, with strong overfit;
- MIST baseline: about `0.5420`.

Those results do not prove that spectrum self-supervision is useless. They show
that the tested DreaMS objectives were not aligned enough with FRIGID's
fingerprint/DLM handoff.

A JEPA-like objective is attractive because it does not force exact raw peak
reconstruction. Instead, the model predicts the latent representation of a
hidden spectrum region from the visible peaks. This should be less sensitive to
raw intensity noise, instrument variation, and missing peaks.

## Method

Training script:

```text
scripts/train_spectral_jepa.py
```

Input spectra are represented as peak tokens:

```text
token_i = [m/z, intensity]
```

Context/target masks are sampled by a mixture of:

- m/z-window masking;
- high-intensity peak masking;
- random peak masking.

Pretraining objective:

```text
context peaks -> context encoder -> predictor -> predicted target latent
target peaks  -> EMA target encoder -> target latent
loss = latent prediction + variance regularization + covariance regularization
```

The target encoder is an exponential-moving-average copy of the context
encoder. This follows the JEPA/BYOL-style stabilization pattern and avoids
training only by raw peak reconstruction.

After pretraining:

1. Freeze the context encoder.
2. Encode full train and validation spectra.
3. Train a supervised fingerprint head on the frozen embeddings.
4. Evaluate validation fingerprints with threshold and top-k sweeps.

## Inputs

Remote host:

```text
spectrum
```

Base DreaMS dataset package:

```text
/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/dreams_fingerprint_head_msg_full_20260629T110040Z
```

Train inputs:

```text
train_dataset/spectra.hdf5
train_dataset/metadata.csv
train_dataset/fingerprints.npz
```

Validation inputs:

```text
val_dataset/spectra.hdf5
val_dataset/metadata.csv
val_dataset/fingerprints.npz
```

## Launch

The Slurm launcher is:

```text
docs/FRIGID_spectral_jepa_pretraining_20260630/run_spectral_jepa.sbatch
```

Submit after the current active GPU jobs clear:

```bash
cd /home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head
mkdir -p runs/slurm_logs
sbatch --dependency=afterany:34 docs/FRIGID_spectral_jepa_pretraining_20260630/run_spectral_jepa.sbatch
```

The dependency keeps this run behind:

- DLM adaptation job `32`;
- DreaMS adapter replacement job `34`.

## Metrics

Primary metric:

```text
best validation mean fingerprint Tanimoto
```

Secondary metrics:

- median fingerprint Tanimoto;
- bit precision, recall, and F1;
- mean predicted active bits;
- best threshold/top-k mode;
- pretraining latent loss trend;
- variance/covariance collapse indicators.

## Decision Gates

This run is a representation gate, not a molecule-generation gate.

Interpretation:

- below `0.25`: not useful compared with previous DreaMS variants;
- `0.25-0.45`: interesting but still below MIST, inspect failure modes;
- `0.45-0.542`: promising representation, but not a replacement;
- above MIST baseline plus `0.005`: candidate for downstream DLM evaluation.

Do not run expensive DLM decoding unless the fingerprint gate is close to MIST
or clearly improves on the previous DreaMS line.

## Expected Artifacts

Remote run directory:

```text
/home/nikolenko/work/Projects/FRIGID_dreams_fingerprint_head/runs/spectral_jepa_pretraining_<UTC>
```

Expected files:

- `train.log`;
- `run_identity.json`;
- `model/run_config.json`;
- `model/pretrain_metrics.json`;
- `model/best_jepa_encoder.pt`;
- `model/last_jepa_encoder.pt`;
- `model/best_fingerprint_head.pt`;
- `model/head_metrics.json`;
- `model/summary.json`;
- `model/val_sweep_metrics.json`;
- `model/val_sweep_metrics.csv`;
- `model/val_predictions.npz`;
- `model/val_prediction_metadata.csv`.

