import pytest

from scripts.train_marlin_spectrum_adaptation import (
    parse_args,
    resolve_clearml_output_uri,
    validate_args,
)


def test_clearml_output_uri_is_disabled_by_default(monkeypatch):
    """SLURM keeps checkpoints on a shared filesystem, so uploading them is
    wasteful; the historical behaviour must stay the default."""
    monkeypatch.delenv("MARLIN_CLEARML_OUTPUT_URI", raising=False)

    assert resolve_clearml_output_uri() is False


def test_clearml_output_uri_is_configurable_for_remote_workers(monkeypatch):
    """A queued worker has its own filesystem, so checkpoints are unreachable
    unless they are uploaded to the files server."""
    monkeypatch.setenv(
        "MARLIN_CLEARML_OUTPUT_URI", "https://files.clearai.innopolis.university"
    )

    assert (
        resolve_clearml_output_uri() == "https://files.clearai.innopolis.university"
    )


def test_clearml_output_uri_ignores_blank_configuration(monkeypatch):
    monkeypatch.setenv("MARLIN_CLEARML_OUTPUT_URI", "   ")

    assert resolve_clearml_output_uri() is False


def _args(monkeypatch, *extra):
    argv = [
        "train_marlin_spectrum_adaptation.py",
        "--checkpoint", "ckpt",
        "--checkpoint-sha256", "abc",
        "--tokenizer", "tokenizer.json",
        "--metadata", "train.csv",
        "--fingerprints", "train.npz",
        "--exclude-inchikeys", "test.csv",
        "--validation-metadata", "val.csv",
        "--validation-fingerprints", "val.npz",
        "--output-dir", "run",
        *extra,
    ]
    monkeypatch.setattr("sys.argv", argv)
    return parse_args()


def test_periodic_evaluation_defaults_to_the_sharded_full_validation_panel(monkeypatch):
    """The 32-spectrum micro panel is retired: its floor is one molecule in 32
    and it swings by up to 0.12 between neighbouring checkpoints."""
    args = _args(monkeypatch)

    assert args.evaluation_manifest.name == "nplib1_val_full396_v1.tsv"
    assert args.evaluation_spectra == 396
    assert args.evaluation_candidates == 8
    assert args.evaluation_shards == 16
    validate_args(args)


def test_held_out_loss_and_selection_stay_off_unless_asked(monkeypatch):
    args = _args(monkeypatch)

    assert args.validation_loss_fraction == 0.0
    assert args.select_best_checkpoint is False
    assert args.selection_patience == 0
    assert args.selection_metric == "candidate_return_rate"


def test_a_full_panel_without_sharding_is_refused(monkeypatch):
    args = _args(monkeypatch, "--evaluation-shards", "1")

    with pytest.raises(ValueError, match="full validation panel needs sharding"):
        validate_args(args)


def test_a_micro_panel_may_still_run_in_one_process(monkeypatch):
    args = _args(
        monkeypatch,
        "--evaluation-shards", "1",
        "--evaluation-spectra", "32",
        "--evaluation-manifest",
        "configs/benchmarks/nplib1_v1/nplib1_val_micro32_v1.tsv",
    )

    validate_args(args)


def test_invalid_held_out_and_selection_settings_are_refused(monkeypatch):
    with pytest.raises(ValueError, match="validation-loss-fraction"):
        validate_args(_args(monkeypatch, "--validation-loss-fraction", "0.9"))
    with pytest.raises(ValueError, match="selection-patience"):
        validate_args(_args(monkeypatch, "--selection-patience", "-2"))
    with pytest.raises(ValueError, match="evaluation-shards must be at least 1"):
        validate_args(_args(monkeypatch, "--evaluation-shards", "0"))


def test_selecting_on_exact_at_one_is_refused_before_the_run_starts(monkeypatch):
    args = _args(
        monkeypatch, "--select-best-checkpoint", "--selection-metric", "exact_top1"
    )

    with pytest.raises(ValueError, match="cannot drive checkpoint selection"):
        validate_args(args)


def test_selecting_on_candidate_return_is_accepted(monkeypatch):
    validate_args(
        _args(
            monkeypatch,
            "--select-best-checkpoint",
            "--selection-patience", "3",
            "--selection-min-delta", "0.01",
        )
    )


def test_the_new_instrumentation_stays_off_unless_asked(monkeypatch):
    """Default behaviour must be byte-for-byte the historical recipe: one global
    gradient norm every 50 steps and no probe."""
    args = _args(monkeypatch)

    assert args.gradient_diagnostics_interval == 0
    assert args.conditioning_probe_interval == 0
    assert args.conditioning_probe_metadata is None
    assert args.conditioning_probe_fingerprints is None
    assert args.conditioning_probe_threshold is None
    validate_args(args)


def test_the_conditioning_probe_refuses_to_run_without_a_fixed_probe_set(monkeypatch):
    """A probe drawn from whatever split was lying around is not comparable
    between runs, which is the whole point of the probe."""
    args = _args(monkeypatch, "--conditioning-probe-interval", "500")

    with pytest.raises(ValueError, match="conditioning-probe-metadata"):
        validate_args(args)


def test_the_conditioning_probe_refuses_a_batch_that_cannot_be_rolled(monkeypatch):
    common = (
        "--conditioning-probe-interval", "500",
        "--conditioning-probe-metadata", "probe.csv",
        "--conditioning-probe-fingerprints", "probe.npz",
    )

    with pytest.raises(ValueError, match="batch-size must be at least 2"):
        validate_args(
            _args(monkeypatch, *common, "--conditioning-probe-batch-size", "1")
        )
    with pytest.raises(ValueError, match="probe-size must be at least 2"):
        validate_args(_args(monkeypatch, *common, "--conditioning-probe-size", "1"))


def test_the_configured_probe_and_gradient_intervals_are_accepted(monkeypatch):
    args = _args(
        monkeypatch,
        "--gradient-diagnostics-interval", "100",
        "--conditioning-probe-interval", "500",
        "--conditioning-probe-metadata", "probe.csv",
        "--conditioning-probe-fingerprints", "probe.npz",
    )

    validate_args(args)
    assert args.conditioning_probe_size == 64
    assert args.conditioning_probe_batch_size == 8
    assert args.conditioning_probe_seed == 0


def test_negative_instrumentation_intervals_are_refused(monkeypatch):
    with pytest.raises(ValueError, match="gradient-diagnostics-interval"):
        validate_args(_args(monkeypatch, "--gradient-diagnostics-interval", "-1"))
    with pytest.raises(ValueError, match="conditioning-probe-interval"):
        validate_args(_args(monkeypatch, "--conditioning-probe-interval", "-1"))


def test_the_recipe_corrections_are_all_off_by_default(monkeypatch):
    """Nothing changes under a run that does not ask for it."""
    args = _args(monkeypatch)

    assert args.lr_schedule == "constant"
    assert args.derive_learning_rate is False
    assert args.learning_rate == 1e-5
    assert args.time_sampling == "per_block_iid"
    assert args.loss_reduction == "block_mean"
    assert args.fp32_forward is False
    assert args.probe_early_stopping_metric is None
    validate_args(args)


def test_the_constant_recipe_records_how_far_it_is_from_the_terminal_rate(monkeypatch):
    from scripts.train_marlin_spectrum_adaptation import resolve_learning_rate

    args = _args(monkeypatch, "--max-steps", "20000")
    plan = resolve_learning_rate(args)

    assert plan["schedule"] == "constant"
    assert plan["peak"] == 1e-5
    assert plan["peak_over_released_terminal"] == pytest.approx(189.76, rel=1e-3)
    assert plan["displacement_bound"] == pytest.approx(0.2)


def test_the_derived_peak_is_recorded_with_its_derivation(monkeypatch):
    from scripts.train_marlin_spectrum_adaptation import resolve_learning_rate

    args = _args(
        monkeypatch,
        "--max-steps", "20000",
        "--lr-schedule", "warmup_cosine",
        "--derive-learning-rate",
    )
    validate_args(args)
    plan = resolve_learning_rate(args)

    assert plan["schedule"] == "warmup_cosine"
    assert plan["peak"] == pytest.approx(3.1647e-7, rel=1e-3)
    assert plan["warmup_steps"] == 1000
    assert plan["floor"] == pytest.approx(5.2697058404552555e-08)
    assert plan["peak_over_released_terminal"] == pytest.approx(6.0, rel=1e-2)
    assert "final decade" in plan["source"]


def test_deriving_a_rate_without_a_schedule_is_refused(monkeypatch):
    args = _args(monkeypatch, "--derive-learning-rate")

    with pytest.raises(ValueError, match="nothing to derive"):
        validate_args(args)


def test_a_peak_below_the_floor_is_refused(monkeypatch):
    args = _args(
        monkeypatch,
        "--lr-schedule", "warmup_cosine",
        "--max-steps", "20000",
        "--learning-rate", "1e-9",
    )

    with pytest.raises(ValueError, match="below --lr-min"):
        validate_args(args)


def test_probe_early_stopping_needs_a_probe(monkeypatch):
    args = _args(
        monkeypatch,
        "--probe-early-stopping-metric", "probe_top1_predicted",
    )

    with pytest.raises(ValueError, match="conditioning-probe-interval"):
        validate_args(args)


def test_probe_early_stopping_is_accepted_with_a_probe(monkeypatch):
    args = _args(
        monkeypatch,
        "--conditioning-probe-interval", "500",
        "--conditioning-probe-metadata", "val.csv",
        "--conditioning-probe-fingerprints", "val.npz",
        "--probe-early-stopping-metric", "probe_top1_predicted",
    )

    assert args.probe_early_stopping_patience == 3
    validate_args(args)
