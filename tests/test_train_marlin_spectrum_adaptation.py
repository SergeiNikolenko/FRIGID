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
