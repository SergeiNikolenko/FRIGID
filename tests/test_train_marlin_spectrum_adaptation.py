from scripts.train_marlin_spectrum_adaptation import resolve_clearml_output_uri


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
