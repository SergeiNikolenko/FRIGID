from inspect import signature
from pathlib import Path

from omegaconf import OmegaConf

from marlin.sampler import MarlinSampler


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_training_config_matches_paper_recipe():
    config = OmegaConf.load(PROJECT_ROOT / "configs/marlin_nplib1.yaml")

    assert config.frigid_warm_start_checkpoint.endswith("/frigid/DLM.ckpt")
    assert config.frigid_warm_start_sha256 == (
        "b6177c2d43448380aba80ff41c01461ea34ca2ca93b213986954c5afb7f0f457"
    )
    assert config.model.block_width == 8
    assert config.model.vocab_size == 1880
    assert config.model.hidden_size == 896
    assert config.model.num_layers == 12
    assert config.model.num_heads == 14
    assert config.model.fingerprint_layer_norm
    assert config.model.frigid_compatible_layer_order
    assert config.model.fingerprint_self_attention_layers == 3
    assert config.training.noise_probability == 0.5
    assert config.training.noise_min_fraction == 0.1
    assert config.training.noise_max_fraction == 0.3
    assert config.optim.learning_rate == 5e-5
    assert config.training.ema_decay == 0.9999
    assert config.loader.batch_size * config.trainer.accumulate_grad_batches == 256
    assert "resume_weights_only_checkpoint" not in config
    assert "filtered_prefix_only" not in config.data

    assert "adaptation" not in config
    assert "layer0_long_residual_scale" not in config.model
    assert "eos_loss_weight" not in config.training
    assert "balanced_token_loss_alpha" not in config.training
    assert "full_sequence_mask_probability" not in config.training
    assert "conditioning_only_steps" not in config.training
    assert "cross_attention_only_steps" not in config.training


def test_inference_defaults_match_paper_recipe():
    parameters = signature(MarlinSampler.generate_ranked_with_stats).parameters

    assert parameters["candidates"].default == 384
    assert parameters["diversity_dropout"].default == 0.3


def test_sampler_defaults_to_deterministic_token_selection():
    parameters = signature(MarlinSampler.__init__).parameters

    assert parameters["sample_tokens"].default is False


def test_oracle_slurm_forwards_eos_boost():
    script = (
        PROJECT_ROOT / "scripts/slurm_marlin_oracle_eval.sbatch"
    ).read_text()

    assert 'EOS_BOOST="${MARLIN_EOS_BOOST:-1.0}"' in script
    assert '--eos-boost "$EOS_BOOST"' in script


def test_faro_gate_keeps_the_paper_batch_and_fresh_frigid_start():
    script = (
        PROJECT_ROOT / "scripts/run_marlin_faro_paper_gate.sh"
    ).read_text()

    assert "trainer.devices=2" in script
    assert "trainer.accumulate_grad_batches=16" in script
    assert "resume_checkpoint=" not in script
    assert "frigid_warm_start_checkpoint=" not in script
    assert "conditioning_only_steps=" not in script
    assert "cross_attention_only_steps=" not in script
    assert "evaluation.interval_steps=" in script
