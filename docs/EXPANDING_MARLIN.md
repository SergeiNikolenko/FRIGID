# Expanding MARLIN

Expanding MARLIN is the explicitly experimental adaptation of discrete
Expanding Generative Flows (EFlow) and Expanding Flow Maps (EFM) from
arXiv:2607.21585 to spectrum-conditioned SAFE generation.

It is not the strict reproduction of MARLIN (arXiv:2607.04774). The strict
block-diffusion model, configuration, checkpoints, and evaluator defaults are
unchanged.

## Architecture

The implementation uses:

- the FRIGID-initialized MARLIN token embedding, Transformer layers,
  fingerprint/mass/isotope conditioner, and tied output projection;
- Gaussian latent token vectors in the full SAFE vocabulary space;
- a cosine per-token insertion schedule with a configurable insertion cutoff;
- the paper's hazard-weighted diagonal insertion objective;
- per-token local denoising times;
- the large-vocabulary decoding-error time warp evaluated by Gauss-Hermite
  quadrature;
- bidirectional transport over the active compacted sequence;
- source- and target-time conditioning in every Transformer layer;
- a learned per-gap insertion-count head and bounded binomial expansion;
- EFlow endpoint denoising and EFM diagonal/semigroup consistency objectives;
- final molecular validity, exact-mass, uniqueness, similarity, formula, and
  exact-connectivity evaluation on the existing NPLIB1 lanes.

BOS and EOS are permanent clean anchors. Molecular SAFE tokens are inserted
only in the gaps between them. This removes the external token-length
regressor and makes output length part of the learned generative process.

## Training stages

### 1. EFlow teacher

The EFlow denoising backbone starts from the pinned FRIGID checkpoint. New time
and insertion modules are randomly initialized. The default experimental
adaptation freezes the backbone for 500 optimizer steps while learning the new
flow-time, modulation, boundary, and insertion modules, then unfreezes the full
network. The insertion head remains detached from the backbone for the first
2,000 steps. New flow modules use the paper learning rate `3e-4`, while the
pretrained backbone is capped at the MARLIN fine-tuning rate `5e-5`. EMA tracks
every parameter in stable order across both stages.

```bash
sbatch scripts/slurm_expanding_marlin_train.sbatch
```

The default configuration follows the sequence experiment in the EFM paper:
Gaussian scale 1.25, insertion cutoff 0.5, learning rate `3e-4`, 2,500 warmup
steps, EMA 0.9999, global batch 512, and 200,000 optimizer steps.
Because the SAFE vocabulary time warp places only about 1% of uniform samples
before cutoff 0.5, small-batch runs stratify early and late times 50/50 and
apply exact importance weights to both objectives. This preserves the
paper's uniform-warped-time target while avoiding insertion batches containing
only zero targets.

### 2. EFM student

The student and frozen teacher must have exactly matching decoder and flow
configurations. The student is initialized from the EFlow EMA weights.

```bash
EXPANDING_STAGE=efm \
EXPANDING_TEACHER_CHECKPOINT=/mnt/netstorage/nikolenko/marlin/runs/expanding/eflow/checkpoints/step=200000.ckpt \
sbatch scripts/slurm_expanding_marlin_train.sbatch
```

EFM batches use the paper's 0.75 diagonal probability, midpoint semigroup
target, adaptive consistency weighting, and direct two-time insertion-count
loss.

### Smoke test

```bash
sbatch scripts/slurm_expanding_marlin_smoke.sbatch
```

The smoke job performs a real optimizer step, writes a loadable checkpoint,
and can publish training scalars to ClearML. It is not evidence of molecular
quality.

## Evaluation

The existing NPLIB1 evaluator accepts expanding checkpoints:

```bash
python scripts/evaluate_marlin_nplib1.py \
  --architecture expanding \
  --expanding-steps 32 \
  --checkpoint /path/to/step=200000.ckpt \
  --tokenizer /path/to/tokenizer.json \
  --metadata /path/to/metadata.csv \
  --fingerprints /path/to/fingerprints.npz \
  --fingerprint-key ground_truth \
  --lane dreams \
  --output-dir /mnt/netstorage/nikolenko/marlin/runs/evaluation/expanding
```

The sampler performs the learned insert→transport process and commits token
identities only at the final step. Exact mass is enforced as a final molecular
acceptance filter because applying prefix constraints during continuous
transport would violate the EFlow/EFM state dynamics.

Report candidate return rate, RDKit validity, mass validity, uniqueness,
internal diversity, formula Top-1/Top-10, exact connectivity Top-1/Top-10, and
target Morgan Tanimoto. Training loss alone is not a success criterion.

## Provenance and limitations

- All expanding runs carry `experimental` and `non-paper-architecture` tags.
- EFlow checkpoints are not interchangeable with strict MARLIN checkpoints.
- EFM training requires a completed EFlow teacher; distilling an untrained
  teacher only makes a faster poor model.
- The upstream EFM repository contained no implementation when this adaptation
  was written. Equations 23–33, 83–87, Algorithms 1–4, and the published
  sequence hyperparameters are the implementation source.
- The original EFM molecular graph benchmark is unconditional QM9 generation;
  it does not establish NPLIB1 spectrum-to-structure quality. That claim must
  come from our held-out evaluation.
