# Diagnostic Summary

This probe was run on `ssh:kolmogorov` with `--profile-generation`,
`max_spectra=1`, `formula_matches=1`, `max_attempts=32`, `batch_size=32`, and
`sigma_lambda=0.0`.

It is diagnostic only. The timing includes CUDA synchronization for profiling.

Observed aggregate:

- `seconds_per_case`: `5.315264701843262`
- `generation_time_percentage`: `96.40539654099905`
- `generation_profile_model_forward_time`: `4.455873496830463`
- `generation_profile_model_forward_backbone_time`: `4.403329693712294`
- `generation_profile_model_forward_build_embeddings_time`: `0.021335973404347897`
- `generation_profile_model_forward_formula_conditioning_time`: `0.0006913663819432259`
- `generation_profile_model_forward_fingerprint_conditioning_time`: `0.0006640199571847916`
- `generation_profile_sampling_step_time`: `0.5147920669987798`
- `generation_profile_decode_time`: `0.11458592861890793`

Interpretation:

- Backbone encoder work dominates the `forward` path.
- Conditioning prep is negligible.
- Sampling/decoding are not the main problem.
- The previous unsynchronized breakdown was misleading; this probe is the first
  one that makes the forward decomposition believable.
