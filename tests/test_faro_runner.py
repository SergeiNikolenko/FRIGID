import subprocess
from pathlib import Path


RUNNER = Path("scripts/run_marlin_faro_spectrum_adaptation.sh")


def _extract_soft_fingerprint_block(script: str) -> str:
    start = script.index("SOFT_FINGERPRINT_ARGS=()")
    end = script.index("test ! -e", start)
    return script[start:end]


def test_faro_runner_is_valid_bash():
    result = subprocess.run(
        ["bash", "-n", str(RUNNER)], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


def test_faro_runner_forwards_soft_fingerprint_when_enabled():
    script = RUNNER.read_text()
    block = _extract_soft_fingerprint_block(script)

    probe = (
        'MARLIN_SOFT_FINGERPRINT=1\n'
        + block
        + 'printf "%s" "${SOFT_FINGERPRINT_ARGS[*]}"\n'
    )
    result = subprocess.run(["bash", "-c", probe], capture_output=True, text=True)

    assert result.stdout == "--soft-fingerprint", result.stderr


def test_faro_runner_omits_soft_fingerprint_by_default():
    script = RUNNER.read_text()
    block = _extract_soft_fingerprint_block(script)

    probe = block + 'printf "%s" "${SOFT_FINGERPRINT_ARGS[*]}"\n'
    result = subprocess.run(["bash", "-c", probe], capture_output=True, text=True)

    assert result.stdout == "", result.stderr


def test_faro_runner_passes_the_soft_fingerprint_args_to_training():
    script = RUNNER.read_text()

    assert '"${SOFT_FINGERPRINT_ARGS[@]}"' in script


def test_faro_runner_pins_materialize_to_the_same_runtime_root_it_reads():
    """materialize_marlin_runtime_inputs.py defaults to a different directory
    than RUNTIME_ROOT, so the runner must export the resolved path before
    calling it or training reads from an empty location."""
    script = RUNNER.read_text()

    export_index = script.index('export MARLIN_RUNTIME_INPUT_ROOT="$RUNTIME_ROOT"')
    materialize_index = script.index(
        "python scripts/materialize_marlin_runtime_inputs.py"
    )

    assert export_index < materialize_index


def test_faro_runner_defaults_to_the_sharded_full_validation_panel():
    """The runner is the recipe: it must not default to the retired micro
    panel, and a full panel is only affordable sharded."""
    script = RUNNER.read_text()

    assert "nplib1_val_full396_v1.tsv" in script
    assert "nplib1_val_micro32_v1.tsv" not in script
    assert '--evaluation-shards "${MARLIN_EVALUATION_SHARDS:-16}"' in script
    assert '--evaluation-spectra "${MARLIN_EVALUATION_SPECTRA:-396}"' in script
    assert '--evaluation-candidates "${MARLIN_EVALUATION_CANDIDATES:-8}"' in script


def test_faro_runner_enables_held_out_loss_and_checkpoint_selection():
    script = RUNNER.read_text()

    assert (
        '--validation-loss-fraction "${MARLIN_VALIDATION_LOSS_FRACTION:-0.05}"'
        in script
    )
    assert '--selection-patience "${MARLIN_SELECTION_PATIENCE:-3}"' in script
    assert '"${SELECTION_ARGS[@]}"' in script

    probe = (
        "SELECTION_ARGS=()\n"
        'if [[ "${MARLIN_SELECT_BEST_CHECKPOINT:-1}" == "1" ]]; then\n'
        "  SELECTION_ARGS+=(--select-best-checkpoint)\n"
        "fi\n"
        'printf "%s" "${SELECTION_ARGS[*]}"\n'
    )
    assert probe.splitlines()[1] in script
    default = subprocess.run(["bash", "-c", probe], capture_output=True, text=True)
    disabled = subprocess.run(
        ["bash", "-c", "MARLIN_SELECT_BEST_CHECKPOINT=0\n" + probe],
        capture_output=True,
        text=True,
    )

    assert default.stdout == "--select-best-checkpoint", default.stderr
    assert disabled.stdout == "", disabled.stderr
