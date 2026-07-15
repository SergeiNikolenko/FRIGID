# MS/MS Encoder Benchmark: Decision Report and Reproducible Protocol

Date: 2026-07-16

## Executive decision

MIST remains the FRIGID production encoder. On the locked 15,325-spectrum
evaluation partition it reached mean fingerprint Tanimoto **0.541498**. The
closest reproduced alternative, the released 512-dimensional MIST encoder
packaged with DiffMS, reached **0.437133**. JESTR, MS2DeepScore 2.0,
SpecEmbedding, and MSBERT reached **0.277733**, **0.227692**, **0.194240**, and
**0.184421**, respectively. No candidate met the required `+0.005` paired gain,
so none proceeded to DLM.

This is a scoped claim, not a field-wide leaderboard claim: MIST is the best
audited released candidate for FRIGID's exact spectrum-to-Morgan-4096 contract.
Retrieval models and formula-free or end-to-end generators optimize different
objectives and remain separate tracks.

This is a completed finite comparison of the audited subset, not a proposal to
rerun DreaMS. The
practical decision is to retain MIST and focus any next experiment on a
fundamentally different direct fingerprint objective or on a separately
evaluated retrieval/reranking interface. The exact public MS2DeepScore 2.0 and
SpecEmbedding checkpoints have now been run. Both fail the quality gate;
MS2DeepScore also overlaps 94.7% of evaluation rows through its released
training MGFs, while SpecEmbedding does not publish training identifiers needed
for a complete overlap audit.

The repository contains an ID-safe evaluator that rejects incompatible rows,
fingerprint dimensions, thresholds, and leakage evidence instead of silently
producing a score. All final predictions and per-spectrum metrics are retained
on `spectrum` with immutable hashes.

## What the task is about

FRIGID currently relies on a spectrum encoder that predicts the 4096-bit
fingerprint consumed by the molecular decoder. The research question is not
whether another model can produce a vector of length 4096. It is whether an
alternative encoder predicts the same ordered Morgan bits more accurately on
the same spectra, without train/evaluation leakage, and whether that gain
survives the DLM decoder.

The task therefore has four separate deliverables:

- inventory and classify available MS/MS encoders;
- lock one comparison contract and one evidence format;
- reproduce the existing MIST baseline and compare candidates pairwise;
- run end-to-end DLM only for encoders that pass the encoder-level gate.

## Evidence inventory

The mandatory [Google report](https://docs.google.com/document/d/11ZbuJ9a2pTOReFiC6dbnvSrhhDP6z3CfL6AtEphUVro/edit)
was inspected, including the Encoder/Decoder, Encoder-only, MS2Molecule, MIST,
DreaMS, and DLM sections. The three linked spreadsheets were also inspected:

- [sheet 1](https://docs.google.com/spreadsheets/d/1L-x3txFGRSIu1ZpJmKG1pz6XO18gmqSWtXtFtU9Mbw8);
- [sheet 2](https://docs.google.com/spreadsheets/d/1UcAeZZN5QzCysSBI50XFd6yoyxZntfjWnbGeFQrLfnY);
- [sheet 3](https://docs.google.com/spreadsheets/d/113L6zVP9mqBJMmxMgJohpWdsdKns3MN3OfiTlrLnDrw).

Those sources established the historical MIST value, the closed DreaMS lines,
the DLM upper bound, and the failed DLM-adaptation checkpoints before new
experiments were selected. The previously supplied canonical benchmark bundle
was found and verified on `spectrum`; no separate live request to Sergey was
made during this run.

The required historical data are already available on `spectrum`; waiting for
an additional handoff is not necessary.

| Item | Evidence |
|---|---|
| MSG labels | `/home/nikolenko/work/Projects/FRIGID/repro_cache/msg/labels.tsv`; 231,104 rows; SHA-256 `feb8c9c58be5ced28c2e370f196638610c013cc656d77eebfbe5d42d215ae8fb` |
| MSG split | `/home/nikolenko/work/Projects/FRIGID/repro_cache/msg/split.tsv`; SHA-256 `5b049fd52855eb491719bd3038fb1643c662fa0071367b1fd36a280b268a21e4` |
| Eligible rows | train 191,216; validation 19,043; test 17,082 |
| MIST checkpoint | `/home/nikolenko/work/Projects/FRIGID/repro_cache/mist_msg.pt`; SHA-256 `09b4e93e8d5d35e7472f4d187d72ca316b125368232bb69148df101ec211b92a` |
| DLM checkpoint | `/home/nikolenko/work/Projects/FRIGID/repro_cache/DLM.ckpt`; SHA-256 `b6177c2d43448380aba80ff41c01461ea34ca2ca93b213986954c5afb7f0f457` |
| Clean historical code | `dreams-fingerprint-head`, commit `5fb8fd62715b9b779761fa945af3f80c2a191081` |
| New baseline export | Slurm job `172`, completed in 3m07s with exit `0:0`; `/home/nikolenko/work/Projects/FRIGID_encoder_benchmark_runs/20260715T182006Z_mist_val` |

The older full-validation prediction NPZ files were no longer present. Metrics
JSON and checkpoints survived, but they are insufficient for paired error
analysis. The completed baseline export restores the missing per-spectrum
evidence and records input and output hashes:

- `fingerprints.npz`: 19,043 rows, 4096 bits, 558 MiB, SHA-256
  `181eff747b01fb15568d9bfc7bdf44f5394a9ed85fc8dce5ee10a1c4ed5c053a`;
- `metadata.csv`: SHA-256
  `91a43b3fda96a5b1310ef18a6c5b63a9de6e62cfdf859ab383b5186f6ce9e8ff`;
- `summary.json`: SHA-256
  `f58f7df65ba4b763a2f69041909d79fa72ba5063dcf209197e62fb90209b8785`.

The new evaluator then reproduced the historical MIST result independently
from the restored arrays:

| Metric | Reproduced value |
|---|---:|
| Mean fingerprint Tanimoto | 0.5420425046 |
| Median fingerprint Tanimoto | 0.5483870968 |
| Molecule-balanced mean Tanimoto | 0.5132696630 |
| Mean false-positive bits | 15.6316 |
| Mean false-negative bits | 21.9029 |
| Train/evaluation structure overlap | 0 / 19,043 |

The evaluation used benchmark code revision
`be9a93c4e08a6a43c78361367450b7ff5c4a0cc8`, 22,746 unique declared training
structures, and completed as Slurm job `173` with exit `0:0`. Key evidence
hashes are:

- ordered spectrum IDs:
  `1a033e1e80e1d52d9bf1c7961229309bf32833c7693f7b0d5c9c1a452f670dd8`;
- `benchmark_summary.json`:
  `0bb5c2a8df6db1574d4e7af4e68c12a8d8d94859cdec1deff221d92ffadfcf78`;
- `per_spectrum_metrics.csv`:
  `c2374f9548947acef0c5ee69143c3151ad58eea1227fbce2b88565391db28021`.

### Prospective decision surface

The final molecule-clustered split and calibration replay completed as Slurm
job `174` with exit `0:0` using code revision
`036567d02eb107ebaa637d0eb8ade36620cd35e9`.

| Partition | Rows | Unique structure clusters | Purpose |
|---|---:|---:|---|
| Calibration | 3,718 | 614 | Threshold selection only |
| Evaluation | 15,325 | 2,458 | Candidate promotion decisions |

The calibration curve selected threshold 0.25 with mean Tanimoto 0.544288.
Threshold 0.30 was second at 0.544126. The untouched evaluation result is the
new prospective MIST baseline:

| Metric | Prospective baseline |
|---|---:|
| Mean fingerprint Tanimoto | **0.5414976816** |
| Median fingerprint Tanimoto | 0.5483870968 |
| Molecule-balanced mean Tanimoto | 0.5129821085 |
| Mean false-positive bits | 15.5486 |
| Mean false-negative bits | 22.0872 |
| Train/evaluation structure overlap | 0 / 15,325 |

Future candidates must use this exact manifest and beat 0.5414976816 by at
least 0.005 with a positive paired cluster-bootstrap lower bound. The legacy
full-validation value 0.5420425046 remains useful only for continuity.

Key prospective hashes:

- partition manifest:
  `29dc9736467a8deb24f8decf93c2ca8ebf6bf56a6193677c7d7d84f32ac94262`;
- evaluation ordered spectrum IDs:
  `4cadc15bcc093e435fb397b2754eb159c1ab1433de7e51a0af8e4e77b3d91b83`;
- threshold calibration curve:
  `54a586c27b960ccd089dc00ca6f28237aedcaba4a64054a65a00e529377e8c34`;
- calibrated benchmark summary:
  `e1641ba5400f184a7fe43283813331c32e15c970f88769c734ecf1619c54f71f`;
- evaluation per-spectrum metrics:
  `1db3a6e5aac13cab807f30adfdb98e80575bc163de72a7dcc5a8a4dfae872325`.

![MIST threshold calibration curve](encoder_benchmark_figures/mist_threshold_calibration.png)

## Final prospective benchmark

All six rows below use the same ordered Morgan radius-2, 4096-bit targets, the
same 3,718-row calibration partition, and the same untouched 15,325-row
evaluation partition. Candidate thresholds were selected only on calibration.
Confidence intervals are paired molecule-cluster bootstrap intervals over
2,000 samples.

| Rank | Encoder | Threshold | Mean Tanimoto | Delta vs MIST | 95% CI for delta | Wins / losses / ties | Decision |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | MIST | 0.25 | **0.541498** | 0 | [0, 0] | baseline | Keep |
| 2 | Released DiffMS MIST-512 | 0.30 | 0.437133 | -0.104364 | [-0.143690, -0.069526] | 5,839 / 9,391 / 95 | Quality gate failed; external pretraining overlap unknown |
| 3 | JESTR + shared frozen probe | 0.975 | 0.277733 | -0.263765 | [-0.279889, -0.248284] | 2,909 / 12,399 / 17 | Not promotion-safe under the official pretraining proxy; quality also failed |
| 4 | MS2DeepScore 2.0 + shared frozen probe | 0.975 | 0.227692 | -0.313805 | [-0.329634, -0.297985] | 1,679 / 13,634 / 12 | Quality and released-training overlap gates failed |
| 5 | SpecEmbedding + shared frozen probe | 0.975 | 0.194240 | -0.347258 | [-0.364820, -0.329236] | 1,386 / 13,921 / 18 | Quality gate failed; external training identifiers unavailable |
| 6 | MSBERT + shared frozen probe | 0.95 | 0.184421 | -0.357076 | [-0.376898, -0.336925] | 1,173 / 14,141 / 11 | Quality gate failed; external GNPS overlap unknown |

![Locked prospective encoder ranking](encoder_benchmark_figures/prospective_encoder_ranking.png)

The DiffMS row is not an end-to-end DiffMS gain. Its 59-tensor encoder state is
bit-for-bit identical to the released `encoder_msg.pt`: it is a smaller MIST
fingerprint encoder packaged inside DiffMS. It is reported here only because it
is a directly compatible released encoder.

### Shared probe and row-lock evidence

MSBERT, JESTR, MS2DeepScore 2.0, and SpecEmbedding used the same frozen-backbone
`LayerNorm -> Linear(d, 4096)` head. Model selection used a structure-disjoint
10% holdout from official train and the locked threshold grid. A fresh head and
optimizer were then retrained from scratch on all 191,216 eligible train rows
for the selected number of epochs: 3 for MSBERT and 29 for JESTR. The candidate
bundles have identical ordered train IDs, InChIKeys, and Morgan targets; their
19,043 validation targets exactly match the MIST reference bundle.

| Candidate | Released encoder evidence | Final probe/prediction evidence | Practical note |
|---|---|---|---|
| MSBERT | source `8e4372abcd93aa3c9d9345527de1d250b3cd0aa8`; 121,580,032 parameters; checkpoint `f6a50e1a5650504370e563a9daf9117b4f4ecd60f5ba508dce2ee9e81f064605` | final probe `dec06c173086c68f4901bd55c665a0c90bbe7f369e8a0e61ec01d7297babee8d`; predictions `1942a79ff4facfd054b58803e57d8435261bc82e995ec06052efca74813d11d5` | MIT code; no separate weight license found; GNPS pretraining overlap remains unknown |
| JESTR | source `a5619c18a85a49171d60ead079684ae667cc0dd0`; released checkpoint `b9f2ccc25ae7710d17d30bfa7cc5ca6e3065962fd0bf24d57af9527d804972fa` | final probe `39247b7460ad3ea935516f6f0341e918ddbf6426c3776c2958c1e8068e32729a`; predictions `3f38919d9a99fe959f593f91b9ee7266f635b3ded70bf648ab70519c58ebda3d` | Current released loader pools official train+valid; checkpoint lacks an embedded exact-run manifest |
| MS2DeepScore 2.0 | source tag `2.5.3`, revision `5de7c58c12b4209bbff5c9dbd4ca47bb70507971`; checkpoint `e7e0c57a5d25bfd328e27d00af9e5559ebc5b85eda1640c61c8436b3497f9821` | predictions `a716f7ec1c40985e59763b594ea0292365c1288a73ca575eea9a6b030ce919d0` | Released training MGFs overlap 14,514/15,325 evaluation rows and 2,280/2,458 structures |
| SpecEmbedding | public Space revision `26665447238100728da3640675927f2c8bfb10cd`; checkpoint `0ca0aa002a0d061a95410f7a4055e82c7fcb428d0ba04b5714ac3a4e7f0f5cca` | predictions `a672a64c6ddb3b17d3ec855f0e35be67cd19ac2e9da959470ddf33faf49dff5a` | Exact public Space preprocessing and unmasked pooling reproduced; published archive has models but no training-identifier manifest |
| Released DiffMS MIST-512 | checkpoint `081885de09513803cdddf8b2230be5482c69de61982edc8f479d276c7cb0a06a`; encoder state `40be750b574c3f8a2cde1e5c5ea72ba9b75778ea566e38ac7efbb61dec14338c` | predictions `270358f6dcfd8c8df27c03fbf976fb1364c3807d80a227f2ff38266fa71b2c7b` | 23,364,644-parameter MIST encoder; all tensors match released `encoder_msg.pt`; not the DiffMS generator |

The JESTR audit uses the strongest official conservative proxy because the
checkpoint contains no embedded training manifest and predates the currently
released `split.pkl`. That proxy contains 25,951 structures and overlaps
15,318/15,325 evaluation rows by first-block InChIKey. The two apparent misses
are standardization differences: exact SMILES overlap is 15,325/15,325. The
training-identifier file has SHA-256 `85596499835ec4742a5110847186af0f6178933066d48933f08df2042c5bc0bd`;
the provenance audit JSON has SHA-256
`98116f86b12234551f338419d9386114730223fe703e207037dc31d37438bd57`.

### Newly released checkpoint audits

MS2DeepScore 2.0 was reproduced from the official 2.5.3 source and 500D model.
Its released positive and negative training MGFs contain 600,158 spectra and
37,623 unique structure blocks. They overlap 14,514/15,325 evaluation rows
(94.71%) and 2,280/2,458 evaluation structures (92.76%). The audit JSON has
SHA-256 `cc9081d1166111397bdfa32961aa1967f98f9b7268eab299f49ba0710fdec20b`.
This overlap is recorded as a failed gate rather than hidden behind an
`unknown` label.

SpecEmbedding was reproduced from the exact public Hugging Face Space revision,
including its top-99 peak selection, precursor token, and unmasked mean pooling.
The associated Figshare archive contains the checkpoint family and analysis
artifacts but no training-identifier manifest, so external overlap remains
unknown. Its low score already fails the quality gate independently of that
missing provenance.

CMSSP was audited but not added to the Morgan ranking. The exact positive-mode
checkpoint has 349,798,784 parameters and changes the embedding of a fixed
spectrum when other spectra in the batch change (`L2=15.4131`). The released
`ProjectionHead` passes a two-dimensional `[batch, projection_dim]` tensor to
`MultiheadAttention`, which interprets the first dimension as one unbatched
sequence. Consequently, there is no batch-invariant per-spectrum embedding to
feed into the shared probe without changing the released model semantics. Its
released GNPS/MassBank data also overlap 5,611/15,325 evaluation rows (36.61%).
The batch audit and overlap audit have SHA-256
`c421363c5972fb04754f612dac9fa735df2265a84d03ddabbea8ab13a24a2ea8`
and `90124446b4ae18a1655da680b2cb0af98e62ccfd5fa1ccd05db80ca4aae90d9b`.

### Final evidence package

Extended joint run:
`/home/nikolenko/work/Projects/FRIGID_encoder_benchmark_runs/20260716T002000Z_extended_encoder_ranking/benchmark`.
Evaluator code revision:
`20c4fd60ee48064b54c91d533212de0781825d4b`.

| Artifact | SHA-256 |
|---|---|
| `benchmark_summary.json` | `e7fb528366859d2c938830596d69e6cb7ed0d23bf290e7d469b2d4fdc849b31c` |
| `aggregate_metrics.csv` | `d4c3e684e1f2e460c4041a9e6935295ee815a927bf67b32fcc2e3a5f500f48b4` |
| `per_spectrum_metrics.csv` | `dfcb9635453b6e3c1a1534c7974d095fe4b51df574c8c82e11353def32f39b55` |
| `paired_deltas.csv` | `604d9a6965ed8f3c7745eb765c0aa5177c8954a325c0b6e4a6c9ab26b74cd6e3` |
| `stratified_metrics.csv` | `58b8104a6b59c05c660a7bb3c8292a73573dd54c224501710f2a569e04a6fc95` |
| `threshold_calibration.csv` | `a4641e178b5618a84052fc1db50a5798bebdd1de9bd386e49fee28ce21ca7d96` |
| `extended_error_analysis.csv` | `175800409488a612d8331e401c0ee87610ce72a3a52bc2fe02b84fe864525c28` |
| `representative_spectra.csv` | `03a065dc5044f3f724b5eeeaaf33f62761cd78ca62b0b6f2a58a368e1fb03a40` |
| `spectrum_characteristic_error_analysis.csv` | `b8d92c568adfdff8be6a58a5173e21e260af1b4ddb7fdd243769bf37bb5afeb9` |
| `spectrum_characteristic_parse_summary.json` | `2c8f3830bd7052b693db7038762f25f5dccad1fc68868ae9fafd400b8d694ee2` |

Reported MSBERT/JESTR and DiffMS latency values were collected with different
timing scopes. They are retained for diagnostics but are not ranked against
one another.

## Error analysis

The global error profile explains why none of the alternatives is a drop-in
replacement:

| Encoder | Mean FP bits | Mean FN bits | Main failure mode |
|---|---:|---:|---|
| MIST | 15.5486 | 22.0872 | Baseline |
| Released DiffMS MIST-512 | 13.3439 | 34.4639 | Systematic under-call: fewer FP bits but 12.38 more FN bits than MIST |
| JESTR probe | 29.8779 | 38.0876 | Both over- and under-prediction; checkpoint is not promotion-safe under the official proxy |
| MS2DeepScore 2.0 probe | 33.4741 | 41.6915 | Both error types are high; released training overlap also prevents promotion |
| SpecEmbedding probe | 23.0612 | 47.1504 | Strong under-call; external training identities unavailable |
| MSBERT probe | 40.5085 | 45.1088 | Broad failure of the linear fingerprint interface |

The released MIST-512 encoder is the only alternative with coherent local
wins, but they are strongly conditional:

| Stratum | Rows | MIST | MIST-512 | Delta |
|---|---:|---:|---:|---:|
| `[M+H]+` | 11,519 | 0.546190 | 0.482513 | -0.063677 |
| `[M+Na]+` | 3,806 | 0.527296 | 0.299789 | -0.227508 |
| Orbitrap | 12,937 | 0.538285 | 0.409710 | -0.128575 |
| QTOF | 2,019 | 0.563629 | 0.582999 | +0.019370 |
| Target bits Q1, 15-51 | 4,052 | 0.493195 | 0.536858 | +0.043663 |
| Target bits Q4, 81-108 | 3,724 | 0.658655 | 0.325157 | -0.333498 |
| Precursor mass Q1, 197.03-405.12 | 3,834 | 0.461046 | 0.530790 | +0.069744 |
| Precursor mass Q4, 761.29-997.49 | 3,798 | 0.653156 | 0.312086 | -0.341070 |
| Peak count Q1, 1-6 | 3,949 | 0.498650 | 0.224749 | -0.273901 |
| Peak count Q4, 33-299 | 3,734 | 0.509667 | 0.690773 | +0.181106 |

Target active-bit density was used only for post-hoc error analysis after all
predictions and thresholds were frozen; it was never supplied to an encoder.
All 15,325 evaluation IDs matched one `.ms` file; precursor mass and peak count
had zero missing values and zero parse failures. The peak-rich gain persists in
separate Orbitrap and QTOF `[M+H]+` strata, so it is a hypothesis-generating
specialist pattern rather than only an instrument-mix artifact. It has no
subgroup bootstrap interval, multiple-comparison correction, or prospective
holdout confirmation. It also does not offset collapse on sparse, high-mass,
dense-target, and sodium-adduct spectra. MIST-512 is therefore at most a
conditional-ensemble hypothesis after external-overlap auditing and a
predeclared subgroup validation, not a production replacement.

Formula composition was used only as a chemistry proxy because MSG does not
contain a structural chemical-class ontology. For MIST-512, deltas versus MIST
were -0.2121 for CHO-only formulas, -0.0568 for CHNO, -0.0250 for formulas
containing P or S, and -0.1178 for halogenated formulas. These are composition
families, not claims about scaffold or functional-group classes.

JESTR is slightly less poor on `[M+Na]+` than `[M+H]+` (0.3150 versus 0.2654),
but the released encoder's exact-SMILES overlap makes that descriptive only.
MSBERT is almost flat across adducts (0.1845 versus 0.1840), consistent with a
generally unsuitable linear fingerprint projection rather than one isolated
data regime.

Representative extremes are retained for every candidate. For example,
MIST-512 scored 1.0000 versus MIST 0.0645 on
`MassSpecGymID0030348` (`C13H22N4O3S`, `[M+H]+`, QTOF), but 0.0769 versus MIST
1.0000 on `MassSpecGymID0051146` (`C12H4Cl2F6N4OS`, `[M+H]+`, QTOF). Individual
examples are diagnostic, while the paired cluster-bootstrap result controls
the decision.

## Existing results, kept in separate lanes

### Encoder-level validation

The historical MIST sweep used 19,043 eligible validation spectra, Morgan
radius 2 with 4096 bits, and selected threshold 0.25.

| Model or intervention | Mean fingerprint Tanimoto | Decision |
|---|---:|---|
| MIST, threshold 0.25 | 0.542043 | Baseline |
| DreaMS frozen head | 0.123805 | Closed |
| DreaMS calibrated | 0.234203 | Closed |
| DreaMS distillation | 0.240354 | Closed |
| DreaMS full fine-tune | 0.2580 | Closed; strong overfit |
| DreaMS staged adapter | 0.232862 | Closed |
| DreaMS spectral JEPA | 0.192590 | Closed |
| MIST + DreaMS residual | 0.542726 | Gain 0.000684; fails 0.005 gate |
| Public SpecEmbedding historical report | 0.3184 | Prediction artifact/checkpoint not found; not independently reproducible |

![Historical fingerprint-encoder results](encoder_benchmark_figures/historical_encoder_metrics.png)

The historical MIST threshold was selected on the same full validation surface
used for reporting. It remains the continuity reference, but it is not the
prospective decision surface. New comparisons must freeze each threshold on a
calibration partition before evaluating the disjoint decision partition.

The surviving internal `SpecEmbedNet` checkpoint must not be confused with the
public contrastive SpecEmbedding model. It borrows parts of that implementation
but directly predicts 4096 fingerprint logits and was trained on 32,768
`custom_buddy` rows. Its retained metrics are 0.194407 validation Tanimoto and
0.159031 test Tanimoto at threshold 0.5. Its training structures overlap the
locked MSG validation surface (77 of 3,072 structures; 495 of 19,043 spectra),
so it fails the leakage gate and is excluded from the ranking. The older 0.3184
report has no surviving prediction bundle or matching checkpoint and is kept
only as an unverified historical note.

### End-to-end test

The full MIST+DLM test used 17,082 eligible spectra and threshold 0.187. Its
mean input-fingerprint Tanimoto was 0.540670; exact top-1/top-10 were
0.109706/0.123932 and generated-structure Tanimoto top-1/top-10 were
0.459838/0.484233. These values must not be mixed with the validation encoder
table because both the split and threshold differ.

A separate 1,400-spectrum paired diagnostic showed the decoder upper bound:

| Fingerprint input | Exact top-1 | Molecular Tanimoto top-1 | Formula success |
|---|---:|---:|---:|
| Ground truth | 0.4879 | 0.8130 | 0.7643 |
| MIST binary | 0.1386 | 0.5677 | 0.6936 |

![DLM outcomes by fingerprint source](encoder_benchmark_figures/dlm_fingerprint_upper_bound.png)

The run was partial (`completed: false`), so it is diagnostic evidence rather
than a final benchmark. The attempted DLM adaptations regressed and remain
closed:

| DLM checkpoint | Ground-truth-fingerprint molecular Tanimoto top-1 | MIST-fingerprint molecular Tanimoto top-1 |
|---|---:|---:|
| Original DLM | 0.3897 | 0.3209 |
| 2,500-step MIST adaptation | 0.3109 | 0.2796 |
| 10,000-step mixed adaptation | 0.3486 | 0.2870 |

No new DLM run was launched for the candidates in the prospective table. This
is a predeclared gate outcome rather than missing compatible evidence: every
candidate failed the encoder-level quality gate, and JESTR additionally failed
the overlap gate.

## Locked benchmark contract

### Data and information boundary

- Train: eligible official MSG train rows only.
- Calibration: a deterministic molecule-clustered partition of official MSG
  validation, used only to freeze thresholds and probe hyperparameters.
- Evaluation: the remaining molecule-disjoint validation partition.
- Final confirmation: official MSG test, exactly once after a model passes the
  validation gate.
- Group identity: first block of the InChIKey.
- Allowed encoder inputs must be declared: peaks, precursor, formula, adduct,
  instrument, and subformula annotations.
- Forbidden inputs: SMILES, InChIKey, ground-truth fingerprint, target active
  bit count, or any statistic derived from evaluation labels.
- External pretraining overlap is `unknown` until audited; it is never silently
  labeled leakage-safe.

### Fingerprint identity

- Type: RDKit Morgan fingerprint.
- Radius: 2.
- Bits: 4096.
- `useChirality`: false.
- Bit numbering must match the reference exactly.

A custom 4096-dimensional vector or a dense embedding is not automatically a
compatible fingerprint. Dense encoders enter the fingerprint track only through
the same train-only probe. They may also enter a separate retrieval/reranking
track, where raw embedding metrics are appropriate.

### Metrics and promotion gate

Primary metric: per-spectrum binary fingerprint Tanimoto.

Required secondary evidence:

- median and molecule-balanced mean Tanimoto;
- soft Tanimoto and BCE;
- bit precision, recall, F1, FP bits, FN bits, and active-bit counts;
- paired wins/losses/ties versus MIST;
- molecule-cluster paired bootstrap 95% interval;
- latency when the exporter supplies it;
- train/evaluation structure overlap status.

A candidate passes the encoder gate only when all conditions hold:

1. paired mean Tanimoto gain is at least 0.005;
2. the lower bound of the molecule-cluster bootstrap interval is above zero;
3. declared training structures do not overlap evaluation structures;
4. the exact prediction bundle, manifest, code revision, checkpoint, and data
   hashes are retained.

Only a passing encoder is connected to DLM. The end-to-end comparison must use
the same formula source, generation budget, seeds, filtering, and ranking.

## Candidate landscape

### Representation taxonomy

- **Direct fingerprint predictors:** MIST, released MIST-512 from DiffMS,
  IDSL_MINT, ms-mole, MetFID, and the unreleased MARLIN DreaMS head. Only the
  two released MIST checkpoints were ready for exact inference without new
  training; MetFID uses incompatible bit semantics.
- **Dense spectrum encoders:** MSBERT, MS2DeepScore, and SpecEmbedding. They
  require a common projection for the existing DLM interface and should also
  be judged separately for retrieval.
- **Contrastive spectrum-molecule encoders:** JESTR, CMSSP, ChemEmbed, CLERMS,
  SpecBridge, MVP, FLARE, and MSAlign. Their native objective is
  alignment/ranking, not ordered Morgan bit prediction.
- **Formula-aware, formula-free, or hybrid generators:** DiffMS, MSFlow,
  MARLIN, FlowMS, MSAnchor, and test-time-tuned spectrum language models.
  These alter more than the encoder and therefore belong in a separate
  end-to-end track.

### Comparative reproducibility table

| Candidate (year) | Architecture / native representation | Training and reported task | Code, weights, license | FRIGID interface / measured compute | Outcome or principal risk |
|---|---|---|---|---|---|
| MIST (2023) | Formula-aware spectrum-to-Morgan predictor | Local MSG checkpoint; fingerprint prediction | MIT code and released weights; separate weight license unknown | Direct Morgan-4096; job `172` took 3m07s for 19,043 rows | Baseline, 0.541498 prospective Tanimoto |
| MSBERT (2024) | Transformer, 512D dense spectrum embedding | GNPS; library matching, reported recall@1 0.7871 on a structure-disjoint GNPS subset | MIT code and released weights; no separate weight license found | Shared 512->4096 probe; 121.6M parameters; export job `175` took 3m11s | Reproduced, 0.184421; GNPS overlap unknown |
| MS2DeepScore 2.0 (2026; original 2021) | Siamese dense similarity encoder, 500D | Released model trained on combined GNPS, MoNA, MassBank, and MSnLib; reported metrics are not Morgan Tanimoto | Apache-2.0 code and Zenodo model/data records | Shared 500->4096 probe; jobs `189`, `191`, `192`, and `194` | 0.227692; released training overlaps 94.7% of evaluation rows |
| JESTR (2025) | Binned-spectrum MLP, 512D joint spectrum-molecule embedding | Candidate ranking on NPLIB1 and MassSpecGym-style data | MIT code and released checkpoints; separate weight license unknown | Shared 512->4096 probe; export job `178` took 1m17s | 0.277733; strongest official proxy has 100% exact-SMILES overlap |
| SpecEmbedding (2025) | Transformer, 512D supervised-contrastive spectrum embedding | GNPS retrieval; reported top-1 hit ratio 0.8173 | Exact public Space source/checkpoint pinned; Space card MIT; Figshare archive CC BY 4.0 | Shared 512->4096 probe; jobs `190` and `194` | 0.194240; published archive omits training identities, so overlap is unknown |
| IDSL_MINT (2024) | Autoregressive sequence model over active fingerprint bits | Direct metabolite fingerprint/structure annotation | MIT package; no general pretrained model matching Morgan-4096 | Requires new exact-target training and autoregressive decoding | Configurable for the contract, but not a released checkpoint comparison |
| ms-mole (2026) | Binned-spectrum MLP with direct Morgan-4096 heads and multiple losses | MassSpecGym retrieval and fingerprint-loss study | MIT code; no pretrained checkpoints | Direct contract match after new training | Official IoU-loss test Tanimoto is 0.1942; useful objective study, not evidence of beating the locked 0.5415 baseline |
| CMSSP (2024) | 2048D cross-modal spectrum-structure embedding | Metabolite identification; reported CASMI and independent-set retrieval gains | Apache-2.0 Hugging Face bundle; exact positive checkpoint pinned | Checkpoint and released data audited in jobs `195` and `196` | Not a well-defined per-spectrum encoder in released code: one spectrum changes with batch neighbors; 36.6% evaluation-row overlap |
| ChemEmbed (2026) | Dense chemical/spectrum embedding | Retrieval and structure-related embedding metrics | Code and Drive-hosted weights; MIT stated in README but no separate LICENSE; weight terms unknown | Shared probe or retrieval track; not run | Artifact and overlap evidence incomplete |
| DiffMS (2025) | Formula-aware MIST encoder plus diffusion generator | Conditional de novo structure generation | MIT code and Zenodo weights; separate weight license unknown | Released MIST-512 direct row evaluated; generator is a separate track | Direct released encoder scored 0.437133; not a DiffMS generator gain |
| MSFlow (2026 preprint) | 512D continuous spectrum representation plus flow decoder | End-to-end molecular generation | Repository says MIT, paper says non-commercial usage; Drive-hosted weight terms unknown | Not compatible with an encoder-only swap; compute not measured | License conflict and separate generator benchmark required |
| CLERMS (2023) | 200D contrastive embedding | Spectrum similarity/retrieval | Public code; pretrained weights and license not confirmed | Shared probe/retrieval track; not run | Reproducibility evidence weaker than selected probes |
| MetFID (2020; public CNN implementation 2022) | Custom 5,618-bit fingerprint | Metabolite identification | Public code and Drive-hosted `.h5` weights; code/weight terms unknown; training partly relies on closed NIST data | Bit identity incompatible with DLM; retraining/new head required | Excluded from direct comparison |
| MSAlign (2026) | DreaMS/ChemBERTa-aligned embedding | Cross-modal retrieval/ranking | Paper is CC BY 4.0; stated code URL currently returns 404; weights/license unknown | Retrieval watchlist only; not currently reproducible | DreaMS-derived direct fingerprint line is already closed |
| SpecBridge (2026) | DreaMS-to-ChemBERTa alignment adapter | Retrieval on MassSpecGym, Spectraverse, and MSnLib | Paper links code and MIT Hugging Face weights, but the code repository is unresolved and the model endpoint returns authorization failure as of 2026-07-16 | Retrieval track only | No currently retrievable immutable implementation bundle |
| MVP / FLARE (2026) | Multiview global alignment / fine-grained peak-node alignment | Candidate retrieval with formula- or mass-based sets | Papers available; no promotion-ready checkpoint plus exact training manifest found | Retrieval track only | Cannot be interpreted as Morgan-4096 replacement evidence |
| MARLIN (2026 preprint) | DreaMS fingerprint head plus formula-free block-diffusion decoder | Formula-free de novo generation on NPLIB1 | Paper states code and models will be released upon publication | Separate end-to-end track | Important current method, but not reproducible yet and not a released encoder candidate |

Primary references:

- MIST: <https://doi.org/10.1038/s42256-023-00708-3>
- MSBERT: <https://doi.org/10.1021/acs.analchem.4c02426>
- MS2DeepScore original: <https://doi.org/10.1186/s13321-021-00558-4>
- MS2DeepScore 2.0: <https://doi.org/10.1038/s41467-026-69083-y>
- JESTR: <https://doi.org/10.1093/bioinformatics/btaf354>
- SpecEmbedding: <https://doi.org/10.1021/acs.analchem.5c02655>
- IDSL_MINT: <https://doi.org/10.1186/s13321-024-00804-5>
- CMSSP: <https://doi.org/10.1021/acs.analchem.4c03724>
- ChemEmbed: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12903953/>
- DiffMS: <https://arxiv.org/abs/2502.09571>
- MSFlow: <https://arxiv.org/abs/2602.19912>
- CLERMS: <https://doi.org/10.1021/acs.analchem.3c00260>
- MetFID: <https://pmc.ncbi.nlm.nih.gov/articles/PMC9547616/>
- MSAlign: <https://openreview.net/forum?id=ZoBAklPA7R>
- ms-mole: <https://arxiv.org/abs/2602.16507>
- SpecBridge: <https://arxiv.org/abs/2601.17204>
- MVP: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12980492/>
- FLARE: <https://doi.org/10.64898/2026.01.27.702086>
- MARLIN: <https://arxiv.org/abs/2607.04774>

## Reproducible evaluator

The evaluator is `scripts/benchmark_encoder_predictions.py`; validation and
metric primitives are in `src/frigid/encoder_benchmark.py`. The deterministic
molecule-cluster partition builder is
`scripts/build_encoder_benchmark_partitions.py`.

New candidate bundle contract:

- `probs`: finite float array `[N, 4096]` in `[0, 1]`;
- `spectrum_ids`: unique one-dimensional IDs for the same rows;
- `inference_seconds`: optional non-negative per-spectrum latency.

Historical bundles without embedded IDs are supported only with an explicit
companion metadata file. The evaluator rejects duplicate, missing, or extra
IDs and reorders candidate rows by identity rather than position.

Build the locked partition manifest once:

```bash
python scripts/build_encoder_benchmark_partitions.py \
  --metadata runs/mist_val/metadata.csv \
  --calibration-fraction 0.2 \
  --seed 42 \
  --output-dir runs/encoder_partitions
```

Then calibrate every model threshold exclusively on the calibration partition
and score exclusively on evaluation:

```bash
python scripts/benchmark_encoder_predictions.py \
  --reference-metadata runs/mist_val/metadata.csv \
  --reference-fingerprints runs/mist_val/fingerprints.npz \
  --reference-model mist=mist_probs \
  --prediction msbert=runs/msbert_val/predictions.npz \
  --selection-manifest \
    runs/encoder_partitions/encoder_benchmark_partitions.csv \
  --calibrate-thresholds \
  --baseline mist \
  --minimum-gain 0.005 \
  --training-identifiers mist=runs/mist_train/inchikeys.txt \
  --external-training-overlap mist=checked \
  --training-identifiers msbert=runs/msbert_train/inchikeys.txt \
  --external-training-overlap msbert=unknown \
  --stratify-column ionization \
  --stratify-column instrument \
  --code-revision <git-commit> \
  --output-dir runs/encoder_comparison
```

The output directory is immutable: the evaluator refuses to overwrite a
non-empty directory. It writes:

- `aggregate_metrics.csv`;
- `threshold_calibration.csv` when calibration is requested;
- `per_spectrum_metrics.csv`;
- `paired_deltas.csv`;
- optional `stratified_metrics.csv`;
- `benchmark_summary.json` with hashes, versions, thresholds, ranking, gate
  decisions, and the ordered spectrum-ID hash.

## Execution record and remaining scope

### Completed for the evaluated audited subset

- Jobs `172`, `173`, and `174` restored MIST per-spectrum predictions,
  reproduced the historical baseline, and created the prospective
  calibration/evaluation split.
- MSBERT export completed as job `175`; the shared two-stage frozen probe and
  immutable prediction bundle were completed and evaluated.
- JESTR export completed as job `178`; the same probe protocol was completed.
  A subsequent source-level audit established complete overlap for the
  strongest official pretraining proxy, so the checkpoint is not
  promotion-safe.
- The released DiffMS MIST-512 encoder export completed as job `183`; its state
  was matched to the official archive and evaluated directly.
- MS2DeepScore 2.0 export/probe completed as job `189`; released training-data
  audits completed as jobs `191` and `192`.
- The exact public SpecEmbedding Space checkpoint export/probe completed as job
  `190`.
- The extended six-model joint evaluator completed as job `194` and checked row
  identity, dimensions, frozen
  thresholds, paired intervals, overlap evidence, categorical strata, and
  per-spectrum errors in one run.
- CMSSP checkpoint-semantics and released-training overlap audits completed as
  jobs `195` and `196`.

### Deferred with an explicit reason

- **IDSL_MINT:** there is no general pretrained checkpoint for this exact
  Morgan-4096 contract. It requires a new direct autoregressive training run,
  so it is not a ready-weight reproduction.
- **ms-mole:** directly supports Morgan-4096 but publishes code rather than
  checkpoints. Its own five-seed IoU-loss result is 0.1942 test Tanimoto, well
  below the locked MIST result even before accounting for the different split.
- **ChemEmbed and CLERMS:** dense/retrieval extensions with incomplete immutable
  weight, license, and exact training-manifest evidence.
- **DiffMS and MSFlow generators:** these change the generator/decoder, so they
  belong to a separate end-to-end benchmark and cannot be credited as a direct
  MIST encoder gain.
- **SpecBridge, MVP, FLARE, and MSAlign:** retrieval/reranking methods rather
  than direct Morgan predictors; no complete promotion-ready artifact bundle
  was available.
- **MARLIN:** the July 2026 formula-free generator includes a DreaMS fingerprint
  head, but its paper explicitly defers code and trained models until
  publication.

### DLM gate outcome

The paired DLM confirmation was not triggered. No clean encoder met the
encoder-level gain rule. Running DLM anyway would spend compute after reading
the answer and would mix encoder and decoder effects, contradicting the locked
methodology.

### Remaining evidence limitations

- MSBERT and released DiffMS MIST-512 still have `unknown` external pretraining
  overlap status.
- Frozen-probe training was deterministic and evaluation uncertainty was
  measured by 2,000 molecule-cluster bootstrap samples, but independent
  multi-seed probe training was not run.
- MSG provides formula but no structural class ontology. Formula composition
  families were analyzed as explicit proxies; scaffold-class claims were not
  invented.
- Current trainer hardening verifies bundle shape and binary targets but does
  not yet require the canonical train-ID/order/target hash at the CLI boundary.
  The final run compensated with a separate cross-bundle hash audit.

### Acceptance status

| Requirement | Status | Evidence or limitation |
|---|---|---|
| Study prior report, required sections, and linked sheets | Complete | Historical values and closed branches were extracted before candidate selection |
| Obtain and lock canonical data and baseline | Complete from prior handoff | Canonical files on `spectrum` were verified by path and SHA-256; no new live contact was needed |
| Survey and classify current encoders beyond MIST/DreaMS | Complete | Representation taxonomy, candidate table, primary references, reproducibility decisions |
| Reproduce multiple selected approaches | Complete for runnable released encoder checkpoints | Released MIST-512, JESTR, MSBERT, MS2DeepScore 2.0, and SpecEmbedding were evaluated; CMSSP was excluded by a reproduced batch-semantics defect and overlap audit |
| Run one common benchmark and retain per-spectrum evidence | Complete | 15,325 evaluation rows, frozen thresholds, paired metrics, immutable hashes |
| Analyze errors and spectrum/chemistry regimes | Complete within available labels | Adduct, instrument, precursor mass, peak count, target density, and formula-composition proxies |
| Validate compatible winners through DLM | Partial; intentionally skipped by the locked gate | No clean candidate passed the encoder gate, so literal end-to-end acceptance remains unfulfilled |
| Demonstrate stability | Partial | The protocol is deterministic and cluster-bootstrap uncertainty is present; no identical rerun or independent training seeds were run |
| Final ranking, recommendation, report, figures, and slides | Complete | MIST retained; artifacts listed above |

## Final interpretation

The benchmark does not support replacing MIST. Released direct fingerprint,
contrastive spectrum-molecule, spectrum-similarity, and spectrum-transformer
routes were tested. All are materially below MIST. JESTR and MS2DeepScore 2.0
also fail explicit released-training overlap checks; SpecEmbedding and MSBERT
retain unknown external overlap.

MIST-512's peak-rich local win is the only observed specialist pattern. It may
justify a prospectively defined conditional ensemble study after
external-overlap auditing, but not a global swap. IDSL_MINT and ms-mole are
train-from-scratch research hypotheses rather than omitted released
checkpoints. Retrieval-oriented dense encoders should be
judged in a separate reranking track rather than forced through a weak linear
fingerprint interface.

Until one of those tracks produces at least `+0.005` mean paired Tanimoto with a
positive cluster-bootstrap lower bound and clean training evidence, MIST stays
in production and DLM remains unchanged.
