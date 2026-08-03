# FRIGID Oracle Fingerprint Ablation Status

## Completed: oracle_run_20

Path:
`/Users/nikolenko/.codex/worktrees/2838/FRIGID/docs/FRIGID_project/FRIGID_oracle_fingerprint_ablation_20260622/oracle_run_20`

This run replaced the MIST-predicted fingerprint with the ground-truth Morgan fingerprint for 20 stratified spectra.

Key result:
- Baseline exact Top-1 / Top-10: 15% / 15%
- Oracle exact Top-1 / Top-10: 45% / 45%
- Baseline Tanimoto Top-10 mean: 0.4686
- Oracle Tanimoto Top-10 mean: 0.8471
- 6 baseline misses became exact hits under oracle fingerprint.
- 11 baseline misses still had no exact match under oracle fingerprint.

Interpretation:
MIST fingerprint quality is a major bottleneck, but not the only bottleneck.

## Completed: oracle_miss11_max500

Path:
`/Users/nikolenko/.codex/worktrees/2838/FRIGID/docs/FRIGID_project/FRIGID_oracle_fingerprint_ablation_20260622/oracle_miss11_max500`

This reran the 11 oracle misses with the oracle fingerprint, `formula_matches=10`, and `max_attempts=500`.

Key result:
- Total spectra: 11
- Exact Top-1 / Top-10: 27.27% / 27.27%
- Tanimoto Top-10 mean: 0.7152
- Baseline exact Top-10 on the same 11: 0%
- Baseline Tanimoto Top-10 mean on the same 11: 0.4368
- Average total generated molecules per case: 244.09

Exact hits recovered by increasing attempts:
- MassSpecGymID0380514
- MassSpecGymID0081883
- MassSpecGymID0058062

Still missed after oracle fingerprint and max500:
- MassSpecGymID0372970
- MassSpecGymID0341761
- MassSpecGymID0347534
- MassSpecGymID0380322
- MassSpecGymID0105109
- MassSpecGymID0152325
- MassSpecGymID0162714
- MassSpecGymID0185133

Interpretation:
Increasing attempts recovered 3 of 11 hard cases, so sampling coverage is a real secondary bottleneck. However, 8 of 11 cases still missed. Several misses already collected 10 formula-matched candidates before reaching 500 attempts, so the next test needs a deeper candidate list, not just more attempts.

## Completed: oracle_miss8_f100_a1000

Remote path:
`/home/nikolenko/work/Projects/FRIGID/repro_runs/oracle_fingerprint_ablation_20260622/oracle_miss8_f100_a1000`

Local path:
`/Users/nikolenko/.codex/worktrees/2838/FRIGID/docs/FRIGID_project/FRIGID_oracle_fingerprint_ablation_20260622/oracle_miss8_f100_a1000`

Settings:
- Spectra: the 8 remaining misses from `oracle_miss11_max500`
- Fingerprint: oracle ground-truth Morgan fingerprint
- Token model: NGBoost MSG token model
- `formula_matches=100`
- `max_attempts=1000`
- `batch_size=16`
- `seed=42`
- shared cross-attention enabled

The script was patched to add:
- `exact_rank`
- `exact_match_top100`
- `tanimoto_top100`
- aggregate `exact_match_top100`
- aggregate `tanimoto_top100_mean`
- aggregate `avg_exact_rank_for_hits`

Key result:
- Total spectra: 8
- Exact Top-1 / Top-10 / Top-100: 0% / 0% / 0%
- Tanimoto Top-10 mean: 0.7123
- Tanimoto Top-100 mean: 0.7123
- Baseline Tanimoto Top-10 mean on the same 8: 0.3409
- Average actual formula matches: 49.875
- Average generated molecules per case: 939.375
- Total elapsed time: 2689.2 seconds
- Exact rank: missing for all 8 cases

Per-case outcome:
- MassSpecGymID0372970: Top-100 miss, Tanimoto 0.6707, 1000 generated, 81 actual formula matches
- MassSpecGymID0341761: Top-100 miss, Tanimoto 0.6484, 1000 generated, 29 actual formula matches
- MassSpecGymID0347534: Top-100 miss, Tanimoto 0.6092, 1000 generated, 69 actual formula matches
- MassSpecGymID0380322: Top-100 miss, Tanimoto 0.7065, 1000 generated, 29 actual formula matches
- MassSpecGymID0105109: Top-100 miss, Tanimoto 0.6966, 1000 generated, 23 actual formula matches
- MassSpecGymID0152325: Top-100 miss, Tanimoto 0.9398, 1000 generated, 43 actual formula matches
- MassSpecGymID0162714: Top-100 miss, Tanimoto 0.5051, 515 generated, 100 actual formula matches
- MassSpecGymID0185133: Top-100 miss, Tanimoto 0.9221, 1000 generated, 25 actual formula matches

Interpretation:
This closes the "maybe it is only Top-10" hypothesis for this hard subset. For these 8 cases, the correct connectivity is not present even in the deeper Top-100 candidate list under oracle fingerprint conditioning. The generator can produce chemically similar molecules, sometimes very similar by Morgan Tanimoto, but it still misses the exact target connectivity. That points to a downstream DLM generation/search/ranking limitation for this subset, not only to the MIST encoder.

## Completed: E26_oracle_fingerprint_balanced_200

Remote path:
`/home/nikolenko/work/Projects/FRIGID/repro_runs/20260618T_spectrum_msg_paper_repro/experiments/E26_oracle_fingerprint_balanced_200`

Local path:
`/Users/nikolenko/.codex/worktrees/2838/FRIGID/docs/FRIGID_project/FRIGID_oracle_fingerprint_ablation_20260622/E26_oracle_fingerprint_balanced_200`

Settings:
- Balanced 200-spectrum subset
- Official `scripts/benchmark_spec2mol.py`
- Oracle fingerprint enabled
- NGBoost token model
- `formula_matches=3`
- `max_attempts=30`
- `batch_size=16`
- `fp_threshold=0.187`

Key result:
- Total spectra: 200
- Exact Top-1 / Top-10: 53% / 53%
- Tanimoto Top-1 / Top-10 mean: 0.8246 / 0.8246
- Formula match success rate: 89%
- Average actual formula matches: 3.84
- Average generated molecules: 26.525
- Wall time: 1210.29 seconds

Interpretation:
On a broader balanced subset, oracle fingerprint gives strong quality, which confirms that MIST fingerprint quality is a major bottleneck in the normal pipeline. But the hard 8-case Top-100 run shows that oracle fingerprint is not sufficient for all cases: some targets remain unreachable or incorrectly ranked/generated by the downstream DLM path.
