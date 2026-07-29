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


def test_faro_gate_keeps_paper_batch_and_supports_exact_continuation():
    script = (
        PROJECT_ROOT / "scripts/run_marlin_faro_paper_gate.sh"
    ).read_text()

    assert "trainer.devices=2" in script
    assert "trainer.accumulate_grad_batches=16" in script
    assert 'RESUME_CHECKPOINT="${MARLIN_RESUME_CHECKPOINT:-}"' in script
    assert 'if [[ -n "$RESUME_CHECKPOINT" ]]; then' in script
    assert '"resume_checkpoint=${RESUME_CHECKPOINT//=/\\\\=}"' in script
    assert '"frigid_warm_start_checkpoint=null"' in script
    assert "conditioning_only_steps=" not in script
    assert "cross_attention_only_steps=" not in script
    assert "evaluation.interval_steps=" in script


def test_faro_requirements_pin_legacy_resolver_conflicts():
    requirements = (
        PROJECT_ROOT / "requirements/faro-paper-gate.txt"
    ).read_text()

    assert "setuptools<81" in requirements
    assert "huggingface-hub==0.36.2" in requirements
    assert "tokenizers==0.13.3" in requirements
    assert "fsspec==2024.2.0" in requirements
    assert "dill==0.3.8" in requirements
    assert "multiprocess==0.70.16" in requirements
    assert "bionemo-moco==0.0.2.1" in requirements


def test_faro_evaluation_isolated_from_training_process():
    script = (
        PROJECT_ROOT / "scripts/run_marlin_faro_evaluation.sh"
    ).read_text()

    assert "python -X faulthandler scripts/evaluate_marlin_nplib1.py" in script
    assert '--clearml-task-id "$TASK_ID"' in script
    assert "MARLIN_EVAL_CHECKPOINT" in script
    assert '--generation-mode "${MARLIN_EVAL_GENERATION_MODE:-block}"' in script
    assert "MARLIN_EVAL_DISABLE_GRAMMAR_MASK" in script
    assert "MARLIN_EVAL_DISABLE_MASS_SHELL" in script


def test_faro_frigid_parity_uses_official_sampler_recipe():
    script = (
        PROJECT_ROOT / "scripts/run_frigid_faro_parity.sh"
    ).read_text()

    assert "scripts/evaluate_frigid_parity.py" in script
    assert '--temperature "${FRIGID_PARITY_TEMPERATURE:-0.8}"' in script
    assert '--randomness "${FRIGID_PARITY_RANDOMNESS:-0.5}"' in script
    assert '--fingerprint-key ground_truth' in script
    assert '--clearml-task-id "$TASK_ID"' in script
