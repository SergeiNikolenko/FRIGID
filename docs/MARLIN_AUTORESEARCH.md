# MARLIN autoresearch campaign

This document records the bounded AutoML harness and the July 29, 2026
architecture/inference campaign. It is evidence for the clean-room
reproduction, not a claim that the paper result has been reproduced.

## Harness contract

The campaign separates five responsibilities:

1. The director proposes one-factor changes.
2. Training runs execute from a committed detached worktree through Slurm.
3. The evaluator is pinned to commit
   `8082c57050b65d4be87f4a403fb97338ce307871`.
4. Content-addressed scorer outputs are cached under
   `/mnt/netstorage/nikolenko/marlin/runs/autoresearch/scorer-cache`.
5. A candidate is kept only after held-out molecular metrics improve without
   regressing candidate return, mass validity, or strict validity.

The scorer uses a fixed held-out validation prefix, ground-truth fingerprint
conditioning as a decoder-oracle diagnostic, block decoding, multinomial token
selection, independent conditioning dropout, strict SAFE decoding, and the
mass-shell constraint. The locked NPLIB1 test split is not used for model
selection.

The staged budget is:

- screen: 4 spectra, 16 candidates, seed 42;
- confirm inference scaling: 4 spectra, 64 candidates, seed 42;
- confirm a model candidate: 4 spectra, 16 candidates, seeds 42, 314159, and
  271828;
- final: the two complete 803-spectrum paper lanes, only after a candidate
  passes the held-out gates.

Changing the data slice, candidate count, seed set, evaluator, or metric weights
creates a new comparison stratum. Results from different strata are not used as
model-promotion evidence.

## July 29 results

All short architecture runs start from the EMA weights of
`safe-gpt-pretrain-v3/checkpoints/step=30000.ckpt`, reset optimizer/loop state,
replay the cached training prefix, and train for 500 optimizer steps.

| Run | Slurm | Held-out budget | Score | Return | Mass validity | Strict validity | Decision |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Legacy checkpoint | — | 4 × 16, seed 42 | 0.003125 | 0 | 0 | 0.062500 | Screen reference |
| Fingerprint LayerNorm | 579/580 | 4 × 16, seed 42 | 0.003906 | 0 | 0 | 0.078125 | Reject: no molecular return |
| FRIGID layer ordering | 581/582 | 4 × 16, seed 42 | 0.000781 | 0 | 0 | 0.015625 | Reject |
| Three fingerprint self-attention layers | 583/584 | 4 × 16, seed 42 | 0.002344 | 0 | 0 | 0.046875 | Reject |
| Legacy checkpoint, scaled inference | 586 | 4 × 64, seed 42 | 0.084554 | 0.25 | 0.0625 | 0.027344 | Keep as inference result |

The 64-candidate run returned one unique mass-valid structure for the first
held-out spectrum. It recovered formula `C17H20N2O2` within 4.49 ppm and reached
Morgan Tanimoto 0.197183 to the target. Exact Top-1 and Top-10 remained zero.
The result proves that generation and mass-shell filtering operate end to end,
but it is far below the paper reference and does not justify locked-test
evaluation.

The independent three-seed 4 × 16 baseline has score 0.046994, candidate return
0.083333, mass validity 0.083333, and strict validity 0.026042. Its seed
variance is high, which is why a one-seed architecture screen cannot promote a
checkpoint.

### FRIGID parity and staged adaptation

The official FRIGID warm start is pinned by SHA-256
`b6177c2d43448380aba80ff41c01461ea34ca2ca93b213986954c5afb7f0f457`.
The parity audit at commit `4701a8a` matched the released fingerprint token
sets within `3.73e-8` and final logits within `1.15e-5`; its machine-readable
artifact is
`/mnt/netstorage/nikolenko/marlin/runs/autoresearch/parity/frigid-parity-4701a8a.json`.

The strict adaptation schedule trains conditioning-only parameters through
step 100, cross-attention and conditioning through step 400, then the full
172M-parameter decoder. At step 1000, teacher-forced evaluation on four held-out
rows produced token Top-1 `0.252874`, Top-10 `0.856322`, and a correct-versus-
shuffled fingerprint NLL gain of `0.4594`. This verifies that the block-causal
path and molecular conditioning are active; it does not establish successful
free generation.

The step-1000 molecular gate (four spectra, 16 candidates, seed 42, EMA,
block decoding, strict SAFE and mass shell) produced validity `0.046875`,
candidate return `0`, mass validity `0`, and uniqueness `0`. Slurm job 603
therefore continues the same exact checkpoint and optimizer state to a bounded
step-5000 ceiling, with molecular gates every 1000 steps. Its ClearML task is
`e2689343ba8f40a5b158c866d12e0cb5`; the run root is
`/mnt/netstorage/nikolenko/marlin/runs/autoresearch/candidates/frigid-staged-dc76457-step5000`.
The locked 803-spectrum test remains forbidden until candidate return and mass
validity are non-zero and survive the three-seed confirmation gate.

FARO capacity is a separate execution concern. Its `/mnt/netstorage` is a
worker-local disk, not the Spectrum NFS. Commit `34eb50a` adds a resumable,
hash-verifying cache bootstrap. ClearML task
`6287d3ded9e2470993d4e467a93d7920` materializes the official Zenodo FRIGID
archive and the pinned 94-file SAFE-GPT snapshot before any FARO training task
is allowed to start.

## Decisions and next experiments

The short architecture upgrades are not promoted. Their lower training losses
did not translate to molecules, and the newly initialized paper modules need a
purpose-built adaptation schedule rather than blind 500-step continuation.

The next model campaign should:

1. preserve the working legacy decoder as the control;
2. adapt all paper modules together from the official hashed FRIGID warm start,
   using a staged freeze/unfreeze schedule and molecular gates at fixed steps;
3. compare 16-candidate screens at identical seeds;
4. confirm only candidates with non-zero return and mass validity across all
   three validation seeds;
5. scale confirmed candidates to 64 and then 384 samples;
6. run the locked 803-spectrum DreaMS and MIST lanes exactly once after the
   held-out confirmation gate passes.

The primary scientific blocker is model quality: local Slurm, checkpointing,
detached execution, immutable evaluation, resumable prediction JSONL, scorer
cache, strict SAFE decoding, and mass-shell candidate filtering all completed
successfully. FARO is optional additional capacity and remains gated on its
independent pinned cache bootstrap; it is not required for the active local
continuation.
