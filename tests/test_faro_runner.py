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
