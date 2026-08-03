# FRIGID Project Work Report

Date: 2026-06-22

## Source Threads

This report consolidates the latest useful answers and adjacent recent work from four Codex threads:

- `019edb13-c2bd-7752-ad25-430bc319049b`: MIST diagnostics, threshold tests, per-bit calibration, and ridge adapter.
- `019eeeaa-1cec-7a12-bbd6-51e9d1912f8f`: oracle/ground-truth fingerprint ablations to separate MIST from DLM bottlenecks.
- `019eeedc-9d7d-7470-9987-c5eb9c8a22be`: oracle-refinement training, ClearML integration, and free-running refinement evaluation.
- `019ed1bb-12f7-7d20-969e-7d6dd2c73e3e`: full-run EDA, interactive Plotly report, and reproduction QA blockers.

## Executive Summary

FRIGID has been run on real MSG data with real checkpoints, and the benchmark path is not fake. The full MSG NGBoost aggregate produced `17,082` observed spectra with Exact Top-1 `10.97%`, Exact Top-10 `12.39%`, Tanimoto Top-10 `0.4842`, MIST Tanimoto `0.5407`, and formula success `91.39%`.

However, this is not yet a clean final paper reproduction. The QA audit found that the manifest expects `17,556` test spectra, so `474` test objects are missing from the aggregate. Missing rows occur in `21/36` shards, all `252/252` shard-data symlinks in the copied run package are broken absolute server links, and `pred_smiles_1` matches the scored `proposal_smiles` in only `57.40%` of rows. If missing rows are counted as failures, Exact Top-1 is `10.67%` and Exact Top-10 is `12.06%`.

The scientific bottleneck picture is now fairly clear:

- MIST/spectrum-to-fingerprint conditioning is a major bottleneck.
- DLM generation/search is also a bottleneck on hard cases, even under oracle fingerprint conditioning.
- ICEBERG refinement technically works after a real CE-key fix, but current added candidates have low value and do not lift exact-match quality.
- Simple MIST-side fixes such as global threshold changes, top-N conditioning, per-bit hard thresholds, and a naive ridge adapter are not enough.
- Oracle-refinement training is technically wired through end-to-end and through ClearML, but the current teacher-forced repair objective does not translate into better free-running molecular generation.

## What Was Tried

### 1. Full MSG Baseline and Reproduction QA

The full MSG base NGBoost aggregate exists locally:

- Local artifact: `/Users/nikolenko/.codex/worktrees/2838/FRIGID/docs/FRIGID_project/FRIGID_experiments_20260618/E28_msg_base_full_ngboost_aggregate`
- QA/EDA package: `/Users/nikolenko/.codex/worktrees/2838/FRIGID/docs/FRIGID_project/FRIGID_spectrum_base_final_20260621`
- Interactive report: `/Users/nikolenko/.codex/worktrees/2838/FRIGID/docs/FRIGID_project/FRIGID_spectrum_base_final_20260621/eda_chemical_full/frigid_chemical_eda_report.html`
- QA sources: `/Users/nikolenko/.codex/worktrees/2838/FRIGID/docs/FRIGID_project/FRIGID_spectrum_base_final_20260621/reproduction_qa_sources`

What worked:

- Full aggregate statistics were produced.
- Chemical EDA and train-overlap analysis were turned into a Plotly HTML report.
- The report was validated in a browser: Plotly charts render, no empty charts, no JavaScript errors.
- A reproducible QA layer was added with `qa_summary.json`, coverage CSVs, symlink status, denominator comparison, proposal-position checks, and source README.

What did not work:

- The package is not clean as a final reproduction artifact because coverage and schema issues remain.
- RDKit-dependent formula/invalid-SMILES checks from the pasted QA audit were not rerun locally because the local Python environment did not have RDKit.

Conclusion:

- The benchmark result is useful evidence for diagnosis, but it needs a clean rerun or repaired aggregation before being presented as final reproduction against the paper.

### 2. ICEBERG Refinement and CE-Key Fix

The key bounded post-fix run is:

- Local artifact: `/Users/nikolenko/.codex/worktrees/2838/FRIGID/docs/FRIGID_project/FRIGID_experiments_20260618/E37_iceberg_scaling_msg_50_r2_cpuiceberg_cefix`
- Remote artifact: `/home/nikolenko/work/Projects/FRIGID/repro_runs/20260622T_iceberg_scaling_msg_50_r2_cpuiceberg_cefix`

What worked:

- A real bug was fixed: collision-energy keys were mismatched as strings such as `"10"` versus numeric values such as `10.0`.
- CE-key warnings dropped from `200` before the fix to `0` after the fix.
- A 50-spectrum two-round CPU-ICEBERG run completed with no critical errors.
- Round 1 and round 2 outputs, timing, final predictions, and aggregate stats were written.

Result:

- Exact Top-1: `16%`
- Exact Top-10: `16%`
- Tanimoto Top-1: `0.4505`
- Tanimoto Top-10: `0.4866`
- MIST Tanimoto: `0.5467`
- Runtime: `1583.20` seconds

Follow-up diagnostic:

- Local artifact: `/Users/nikolenko/.codex/worktrees/2838/FRIGID/docs/FRIGID_project/FRIGID_experiments_20260618/E38_a37_ranking_refinement_analysis`
- A37 oracle-best Tanimoto improved only from `0.4819` to `0.4977`.
- Round 2 added a new exact target in only `1/50` cases.
- Exact target was present in only `8/50` round-2 candidate sets.
- The mean best Tanimoto among added candidates was `0.4420`.

Conclusion:

- The CE-key fix was correct and necessary, but ICEBERG quality did not improve enough. The current round-2 candidate pool is weak, so a simple reranker over the current pool is unlikely to close the gap alone.

### 3. MIST Threshold, Rank, Calibration, and Adapter Work

The MIST work tested whether the fingerprint-conditioning signal could be improved cheaply before running DLM.

#### A39: MIST Rank Diagnostics

- Local artifact: `/Users/nikolenko/.codex/worktrees/2838/FRIGID/docs/FRIGID_project/FRIGID_experiments_20260618/E39_mist_rank_balanced_200`
- Remote clean run: `/home/nikolenko/work/Projects/FRIGID/repro_runs/20260622T_a39b_mist_rank_balanced_200`
- Scope: 200 balanced MSG test spectra, MIST-only, no DLM, no ICEBERG.

Key findings:

- Thresholds `0.187`, `0.20`, and `0.25` are nearly tied by bit-level F1.
- Top-64 true-bit recall: `0.6458`
- Top-128 true-bit recall: `0.7452`
- Top-256 true-bit recall: `0.8124`
- Top-768 true-bit recall: `0.8997`
- Top-1024 true-bit recall: `0.9185`

Conclusion:

- Small top-N conditioning loses too many true target bits.
- Reaching about 90% mean recall requires hundreds of active bits, which is too noisy for the current DLM conditioner.

#### A40: Per-Bit Threshold Calibration

- Local artifact: `/Users/nikolenko/.codex/worktrees/2838/FRIGID/docs/FRIGID_project/FRIGID_experiments_20260618/E40_mist_per_bit_calibration_balanced_200`
- Remote artifact: `/home/nikolenko/work/Projects/FRIGID/repro_runs/20260622T_a40_mist_per_bit_calibration_balanced_200`
- Calibration: 5,000 MSG validation spectra.
- Evaluation: fixed balanced200 test subset.

Result:

- Global threshold `0.187`: mean MIST Tanimoto `0.5162`
- Global threshold `0.25`: mean MIST Tanimoto `0.5218`
- Per-bit F1 thresholds: mean MIST Tanimoto `0.4992`

Conclusion:

- Independent per-bit hard thresholds are worse than simple global thresholds. This should not be run downstream through DLM.

#### A41: Ridge Adapter

- Local artifact: `/Users/nikolenko/.codex/worktrees/2838/FRIGID/docs/FRIGID_project/FRIGID_experiments_20260618/E41_mist_ridge_adapter_balanced_200`
- Remote artifact: `/home/nikolenko/work/Projects/FRIGID/repro_runs/20260622T_a41_mist_ridge_adapter_balanced_200`
- Code: `.thoughts/scripts/mist_ridge_adapter.py` in the MIST worktree.
- Calibration: 2,000 MSG validation spectra.
- Evaluation: fixed balanced200 test subset.

Result:

- Best raw baseline: `raw_threshold_0.25`, mean MIST Tanimoto `0.5218`.
- Best ridge variant: `ridge_alpha_100_threshold_0.3`, mean MIST Tanimoto `0.2830`.

Conclusion:

- A naive full 4096-to-4096 ridge regression adapter severely degrades fingerprint quality. A useful learned adapter needs a sparse multi-label objective or a different representation, such as BCE/logistic/MLP residual heads or DreaMS-style spectrum embeddings with a supervised 4096-bit head.

### 4. Oracle Fingerprint Ablations

The oracle fingerprint work tested what happens if DLM receives the true target Morgan fingerprint instead of the MIST-predicted fingerprint.

Main local package:

- `/Users/nikolenko/.codex/worktrees/2838/FRIGID/docs/FRIGID_project/FRIGID_oracle_fingerprint_ablation_20260622`

#### 20-Case Stratified Pilot

- Local artifact: `/Users/nikolenko/.codex/worktrees/2838/FRIGID/docs/FRIGID_project/FRIGID_oracle_fingerprint_ablation_20260622/oracle_run_20`
- Summary: `/Users/nikolenko/.codex/worktrees/2838/FRIGID/docs/FRIGID_project/FRIGID_oracle_fingerprint_ablation_20260622/oracle_run_20_summary.md`

Result:

- Baseline Exact Top-1 / Top-10: `15% / 15%`
- Oracle Exact Top-1 / Top-10: `45% / 45%`
- Baseline Tanimoto Top-10 mean: `0.4686`
- Oracle Tanimoto Top-10 mean: `0.8471`

Conclusion:

- MIST fingerprint quality is a major bottleneck. With perfect fingerprint conditioning, DLM recovers many more exact targets.

#### 11 Oracle-Miss Cases, Higher Attempts

- Local artifact: `/Users/nikolenko/.codex/worktrees/2838/FRIGID/docs/FRIGID_project/FRIGID_oracle_fingerprint_ablation_20260622/oracle_miss11_max500`

Result:

- 3 of 11 previously missed hard cases became exact hits.
- Exact Top-1 / Top-10: `27.27% / 27.27%`
- Tanimoto Top-10 mean: `0.7152`

Conclusion:

- Sampling/coverage is a real secondary bottleneck for some cases.

#### Remaining 8 Hard Cases, Top-100

- Local artifact: `/Users/nikolenko/.codex/worktrees/2838/FRIGID/docs/FRIGID_project/FRIGID_oracle_fingerprint_ablation_20260622/oracle_miss8_f100_a1000`

Result:

- Exact Top-1 / Top-10 / Top-100: `0% / 0% / 0%`
- Tanimoto Top-100 mean: `0.7123`
- Average generated molecules per case: `939.375`
- Correct exact rank was missing for all 8 cases.

Conclusion:

- For the hardest subset, the target connectivity is not present even in a deeper Top-100 list under oracle fingerprint conditioning. This is DLM generation/search limitation, not only a MIST issue.

#### Balanced-200 Oracle Fingerprint Run

- Local artifact: `/Users/nikolenko/.codex/worktrees/2838/FRIGID/docs/FRIGID_project/FRIGID_oracle_fingerprint_ablation_20260622/E26_oracle_fingerprint_balanced_200`

Result:

- 200 spectra
- Exact Top-1 / Top-10: `53% / 53%`
- Tanimoto Top-1 / Top-10 mean: `0.8246 / 0.8246`
- Formula success: `89%`

Conclusion:

- On a broad balanced slice, oracle fingerprint produces a very large quality gain. This confirms that the normal pipeline is strongly limited by the spectrum-to-fingerprint stage.

### 5. Oracle-Refinement Training and ClearML

The oracle-refinement work attempted a learned repair/refinement path: keep a matched fragment and train DLM to repair the bad candidate toward the true molecule.

#### End-to-End Smoke

- Local artifact: `/Users/nikolenko/.codex/worktrees/2838/FRIGID/docs/FRIGID_project/FRIGID_oracle_refinement_e2e_20260622T124305Z`
- Remote artifact: `/home/nikolenko/work/Projects/FRIGID/repro_runs/oracle_refinement_e2e_20260622T124305Z`
- Commit mentioned in the thread: `a6771fd2a2701d58c01a1112bb07569ec78a93a2`

What worked:

- Split-safe train/val oracle traces were built without test leakage.
- DLM checkpoint loaded.
- Two optimizer steps ran on A100.
- Checkpoint was written.
- Sampler loaded the fine-tuned checkpoint and produced refinement generations.
- A previous Lightning batch-transfer blocker was fixed by moving only tensors to device and leaving `formula: list[str]` as metadata.

Smoke metrics:

- Val smoke: 4 traces x 2 samples.
- Exact: `0.0`
- Mean candidate Tanimoto: `0.0397`
- Mean refined Tanimoto: `0.0463`

Conclusion:

- The end-to-end path works technically, but this smoke is not quality evidence.

#### ClearML 2,000-Step Run

- Local artifact: `/Users/nikolenko/.codex/worktrees/2838/FRIGID/docs/FRIGID_project/FRIGID_oracle_refinement_train_clearml_20260622T130906Z`
- ClearML task: `64a2505e2c1e4309aa73fbff06206221`

Result:

- 2,000 steps, batch 4, A100.
- Mean candidate Tanimoto: `0.0414`
- Mean refined Tanimoto: `0.0568`
- Exact: `0.0`

Conclusion:

- ClearML integration and training worked. Quality signal was weak and still not a useful model.

#### Big ClearML Run

- Local artifact: `/Users/nikolenko/.codex/worktrees/2838/FRIGID/docs/FRIGID_project/FRIGID_oracle_refinement_big_clearml_20260622T160222Z`
- Remote artifact: `/home/nikolenko/work/Projects/FRIGID/repro_runs/oracle_refinement_big_clearml_20260622T160222Z`
- ClearML task: `1ec16fea256d4582ae8234d9d7e6c3ba`
- ClearML URL: `https://app.clearai.innopolis.university/projects/ff2caaabaca64025aca4de5cabd2e798/experiments/1ec16fea256d4582ae8234d9d7e6c3ba/output/log`

Dataset:

- Train candidate spectra: `5000`
- Train oracle traces: `47,875`
- Val candidate spectra: `1000`
- Val oracle traces: `8,928`

Training:

- Init checkpoint: `DLM.ckpt`
- Steps: `10,000`
- Batch size: `8`
- GPU: A100 80GB
- Precision: `bf16-mixed`
- Training time: about `1080.9` seconds
- Final checkpoint: `/home/nikolenko/work/Projects/FRIGID/repro_runs/oracle_refinement_big_clearml_20260622T160222Z/training/checkpoints/oracle-refinement-step=10000.ckpt`

Validation eval:

- 256 val traces, 505 predictions.
- Exact match: `0.0`
- Mean candidate Tanimoto: `0.0481`
- Mean refined Tanimoto: `0.0316`

Train-slice diagnostic:

- 128 train traces, 256 predictions.
- Exact match: `2.34%`
- Mean candidate Tanimoto: `0.3836`
- Mean refined Tanimoto: `0.3377`

Conclusion:

- Training was real and reproducible, but the current refinement method is scientifically negative. The model learns a supervised token objective, but free-running sampling does not reliably preserve useful structure and worsens molecules on average.

## Current Conclusions

1. The project is past basic bring-up. Real MSG data, real model checkpoints, full-run aggregation, oracle ablations, ICEBERG diagnostics, and training runs all exist.

2. The current full MSG result should be treated as diagnostic, not final reproduction. Coverage and schema blockers need to be fixed before paper-comparison claims.

3. MIST conditioning is the biggest currently proven bottleneck. Oracle fingerprint runs move Exact Top-10 from `15%` to `45%` on a stratified 20-case pilot and to `53%` on balanced200.

4. DLM generation/search is also limiting. The hard 8-case oracle run found `0/8` exact even in Top-100, despite much higher structural similarity.

5. ICEBERG had a real bug and that bug was fixed, but the current refinement candidate policy is not enough. More brute-force ICEBERG is low priority unless the proposal quality changes.

6. Simple MIST tuning is exhausted for now. Threshold sweeps, top-N, per-bit thresholds, and ridge adaptation do not solve the fingerprint bottleneck.

7. Oracle-refinement needs a different formulation. More steps alone are not enough; the next work should focus on constrained decoding, locked-fragment preservation, formula validity, candidate-similarity reranking, or a fundamentally better repair objective.

## Recommended Next Steps

1. Fix the benchmark reproducibility layer first:
   - repair or rerun missing shards;
   - remove broken absolute symlinks from the delivered package;
   - make aggregate denominator explicit;
   - align `predictions.csv` schema with scored `proposal_smiles`;
   - rerun RDKit formula/invalid-SMILES checks in a proper RDKit environment.

2. Do not run another large current-MIST top-N, per-bit threshold, or ridge downstream DLM experiment. The MIST-only diagnostics already show those variants are weak.

3. Build a real learned spectrum-to-fingerprint improvement:
   - BCE/logistic/MLP residual adapter over MIST probabilities;
   - low-rank residual head;
   - DreaMS-style spectrum embedding plus supervised 4096-bit Morgan head;
   - evaluate MIST-only on balanced200 before any DLM generation.

4. For DLM hard cases, test generation constraints directly:
   - preserve locked fragments during sampling;
   - enforce formula/validity earlier;
   - rerank by candidate similarity and constraints;
   - separately measure whether exact connectivity is generated before ranking.

5. Keep analysis stratified by chemical class and cluster. The EDA shows the failures are not uniform; macrocycles, peptide/depsipeptide-like structures, glycosides, and natural-product-like clusters need separate diagnosis.
