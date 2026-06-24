# FRIGID MSG Base Reproduction QA Sources

This folder contains locally recomputed QA evidence for the copied FRIGID MSG base run preserved under `docs/FRIGID_project/`.

## Key Verified Blockers

- Manifest test samples: 17556
- Aggregate detailed rows: 17082
- Missing test objects: 474 (2.70%)
- Shards with missing rows: 21 / 36
- Reported completed shards: 36
- Reported Exact Top-1 / Top-10: 10.97% / 12.39%
- Exact Top-1 / Top-10 if missing objects are failures: 10.67% / 12.06%
- Broken symlinks: 252 / 252
- `proposal_smiles == pred_smiles_1`: 9805 / 17082 (57.40%)
- Unique target SMILES: 3076 over 17082 spectra
- `data/len.pk not found` warnings: 38

## Files

- `qa_summary.json`: compact machine-readable summary.
- `shard_coverage.csv`: expected vs observed rows by shard.
- `missing_test_spectra.csv`: missing MassSpecGym IDs by shard.
- `symlink_status.csv`: symlink target and broken-link status.
- `proposal_position_in_predictions.csv`: where scored `proposal_smiles` appears in `pred_smiles_1..10`.
- `metric_denominator_comparison.csv`: reported observed denominator vs manifest denominator.
- `micro_vs_macro_metrics.csv`: per-spectrum vs per-target-SMILES metrics.

RDKit-dependent formula and invalid-SMILES checks from the pasted audit were not rerun locally because RDKit is not available in the current local Python environment.
