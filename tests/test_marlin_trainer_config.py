import ast
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from scripts import train_marlin


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_gradient_clip_hydra_default_override_and_validation():
    config_dir = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        default = compose(config_name="marlin_nplib1")
        disabled = compose(
            config_name="marlin_nplib1",
            overrides=["trainer.gradient_clip_val=0"],
        )
        invalid = compose(
            config_name="marlin_nplib1",
            overrides=["trainer.gradient_clip_val=-0.1"],
        )

    assert default.trainer.gradient_clip_val == pytest.approx(1.0)
    assert train_marlin.validate_gradient_clip_val(default) == pytest.approx(1.0)
    assert train_marlin.validate_gradient_clip_val(disabled) == 0.0
    with pytest.raises(ValueError, match="trainer.gradient_clip_val"):
        train_marlin.validate_gradient_clip_val(invalid)


def test_lightning_trainer_receives_validated_gradient_clip_value():
    tree = ast.parse((PROJECT_ROOT / "scripts/train_marlin.py").read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "main"
    )
    assignment = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "gradient_clip_val"
            for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Name)
    assert assignment.value.func.id == "validate_gradient_clip_val"

    trainer_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "L"
        and node.func.attr == "Trainer"
    )
    keywords = {keyword.arg: keyword.value for keyword in trainer_call.keywords}
    gradient_clip = keywords["gradient_clip_val"]
    assert isinstance(gradient_clip, ast.Name)
    assert gradient_clip.id == "gradient_clip_val"
