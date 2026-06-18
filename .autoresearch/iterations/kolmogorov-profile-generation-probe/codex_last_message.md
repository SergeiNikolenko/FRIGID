# Diagnostic Summary

Generation profiling was run on `ssh:kolmogorov` with one GPU, 2 spectra,
`batch_size=32`, `formula_matches=1`, `max_attempts=32`, and
`sigma_lambda=0.0`.

This is not a comparable candidate run because `--profile-generation` adds CUDA
synchronization for measurement.

Observed aggregate profile:

- `seconds_per_case`: `4.7610132694244385`
- `generation_time_percentage`: `97.83771326722832`
- `total_generation_profile_model_forward_time`: `8.801494488492608`
- `total_generation_profile_sampling_step_time`: `0.24337605107575655`
- `total_generation_profile_decode_time`: `0.23280819039791822`
- `total_generation_profile_conditioning_setup_time`: `0.01683588419109583`
- `avg_generation_profile_num_steps`: `118.0`
- `avg_generation_profile_sequence_length`: `120.0`

Interpretation: the measured hot path is model forward across diffusion steps,
not RDKit post-processing, decode, or conditioning setup.
