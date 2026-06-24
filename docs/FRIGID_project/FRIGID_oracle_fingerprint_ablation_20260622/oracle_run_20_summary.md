# FRIGID Oracle Fingerprint Ablation, 20-case Pilot

Source run: `/home/nikolenko/work/Projects/FRIGID/repro_runs/oracle_fingerprint_ablation_20260622/oracle_run_20`

Local copy: `/Users/nikolenko/.codex/worktrees/2838/FRIGID/docs/FRIGID_project/FRIGID_oracle_fingerprint_ablation_20260622/oracle_run_20`

## Setup

- Sample size: 20 spectra, stratified across MIST Tanimoto bins and failure modes.
- Baseline: completed MSG FRIGID-base NGBoost full run.
- Oracle intervention: replace the MIST-predicted fingerprint with the ground-truth Morgan fingerprint computed from target SMILES.
- Generation settings: shared cross-attention, NGBoost token model, `formula_matches=10`, `max_attempts=100`, `batch_size=16`, `sigma_lambda=3.0`.
- This is an upper-bound ablation because the same oracle fingerprint is used for DLM conditioning and formula-candidate ranking.

## Headline Result

| Metric | Baseline sample | Oracle fingerprint | Delta |
|---|---:|---:|---:|
| Exact Top-1 | 15.0% | 45.0% | +30.0 pp |
| Exact Top-10 | 15.0% | 45.0% | +30.0 pp |
| Tanimoto Top-10 mean | 0.4686 | 0.8471 | +0.3784 |

## Interpretation

The oracle fingerprint improves both exact recovery and structural similarity sharply. This supports the hypothesis that MIST-predicted fingerprint quality is a major bottleneck.

The improvement is not complete. Eleven baseline misses remained misses under oracle fingerprint, although many became high-Tanimoto near misses. This means the DLM generation, formula matching, candidate diversity, and ranking path still impose a second bottleneck for exact recovery.

## Case Movement

- 6 baseline Top-10 misses became oracle Top-10 exact hits.
- 3 baseline exact hits stayed exact under oracle.
- 11 baseline misses stayed non-exact under oracle.

## Notable Failure Pattern

Some oracle non-exact cases had very high Top-10 Tanimoto but still missed exact connectivity. Several of these also had low actual formula-matched counts, which suggests that exact recovery is also limited by generation coverage and formula-match candidate diversity, not only by MIST.

## Files

- `oracle_aggregate_statistics.json`
- `comparison_vs_baseline.csv`
- `oracle_detailed_results.csv`
- `oracle_predictions.csv`
- `oracle_run_config.yaml`
- `oracle_run_20.log`
