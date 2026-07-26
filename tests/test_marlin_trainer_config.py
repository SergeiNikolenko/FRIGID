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


def test_tiny16_capacity_gate_has_bounded_reproducible_contract():
    config_dir = str(PROJECT_ROOT / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(
            config_name=(
                "marlin_frigid_distilled_tiny16_short_action_ce_balanced_"
                "cyclic_all_decoder_layer0_residual"
            )
        )

    assert config.resume_checkpoint is None
    assert str(config.resume_weights_only_checkpoint).endswith(
        "layer0-residual-f5aa00b-resume256-to1024/checkpoints/step=1024.ckpt"
    )
    assert config.adaptation.mode == "frigid_distilled_marlin"
    assert config.adaptation.stage.startswith("diagnostic-tiny16-short-")
    assert config.data.metadata_csv_sha256 == (
        "0e7dcf36a25fa568e2d213c7000f344ac426d986d2e370b46947480bad777c86"
    )
    assert config.loader.batch_size == 16
    assert config.loader.num_workers == 0
    assert config.model.layer0_long_residual_scale == pytest.approx(1.0)
    assert config.optim.learning_rate == pytest.approx(5.0e-5)
    assert config.trainer.max_steps == 512
    assert config.trainer.accumulate_grad_batches == 1
    assert config.training.molecular_validation_interval == 128
    assert config.training.molecular_validation_samples == 16
    assert config.training.molecular_validation_candidates == 1
    assert config.training.molecular_validation_use_ema is False
    assert str(config.output.root).startswith(
        "/mnt/netstorage/nikolenko/marlin/runs/"
    )
    assert "not-strict-reproduction" in config.tracking.clearml.tags
    assert "capacity-scale-gate" in config.tracking.clearml.tags
