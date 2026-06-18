# Diagnostic Probe: num_tokens_unmask=2

This run is a one-spectrum comparative probe for the generation hot path on
Kolmogorov.

Observed result:

- `seconds_per_case`: `5.055677652359009`
- `tanimoto_top1`: `0.39393940567970276`
- `formula_success`: `1.0`
- `avg_unique_valid_smiles`: `4.0`
- `generation_time_percentage`: `95.6396139428869`

Comparison point:

- Earlier `num_tokens_unmask=1` profile: `4.7610132694244385 sec/case`,
  `tanimoto_top1 = 0.5888278484344482`, `formula_success = 1.0`

Conclusion:

`num_tokens_unmask=2` regressed on both throughput and quality versus the
earlier `num_tokens_unmask=1` probe. Keep it as a negative diagnostic signal,
not a promoted optimization.
