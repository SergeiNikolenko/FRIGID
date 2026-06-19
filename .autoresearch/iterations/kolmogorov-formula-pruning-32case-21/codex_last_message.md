Adaptive interpolation run with `formula_pruning_chunk_size=21` on Kolmogorov.

Result: `seconds_per_case=3.9312159046530724`, `tanimoto_top1=0.5091927908360958`,
`formula_success=0.90625`.

This point is interesting because it improves Tanimoto relative to the 20-chunk
run, but it drops formula success below 1.0. That makes it a negative signal for
naive interpolation and a reason to move toward adaptive stopping instead of
blindly probing more fixed chunk sizes.
