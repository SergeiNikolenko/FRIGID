import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_slurm_launchers_use_project_runtime_and_tls_preflight():
    launchers = list((PROJECT_ROOT / "scripts").glob("slurm_marlin*.sbatch"))
    assert launchers
    for path in launchers:
        assert "FRIGID/.venv" not in path.read_text(), path
    for name in ("slurm_marlin_train.sbatch", "slurm_marlin_train_smoke.sbatch"):
        script = (PROJECT_ROOT / "scripts" / name).read_text()
        assert 'source "$CODE/scripts/marlin_runtime_env.sh"' in script
    train_script = (PROJECT_ROOT / "scripts/slurm_marlin_train.sbatch").read_text()
    assert '${RESUME_CHECKPOINT//=/\\\\=}' in train_script
    assert "++data.metadata_csv=" in train_script
    assert "++data.metadata_csv_sha256=" in train_script
    assert "MARLIN_METADATA_CSV_SHA256 is required" in train_script


def test_clearml_does_not_capture_its_own_transport_errors():
    tree = ast.parse((PROJECT_ROOT / "scripts" / "train_marlin.py").read_text())
    task_init = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "Task"
        and node.func.attr == "init"
    )
    keywords = {keyword.arg: keyword.value for keyword in task_init.keywords}
    assert isinstance(keywords["auto_connect_streams"], ast.Constant)
    assert keywords["auto_connect_streams"].value is False
