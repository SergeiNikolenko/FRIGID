import json
from types import SimpleNamespace

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

from marlin.conditioning import MarlinConditioner, SparseFingerprintEncoder
from marlin.grammar import SafeGrammarMask
from marlin.mass_shell import MassShellConstraint, MassShellState
from marlin.model import (
    MarlinDecoder,
    MarlinDecoderConfig,
    block_causal_attention_mask,
    two_stream_attention_mask,
)
from marlin.noise import symmetric_fingerprint_noise
from marlin.isotopes import theoretical_isotope_ratios
from marlin.sampler import MarlinSampler
from marlin.token_properties import build_token_property_table, token_properties
from marlin.tokenizer import load_safe_tokenizer
from marlin.training import (
    MarlinCollator,
    MarlinLightningModule,
    MarlinMolecularValidationCallback,
    MarlinTrainingFilter,
)
from marlin.warm_start import (
    _copy_attention,
    _copy_embeddings,
    _copy_fingerprint_encoder,
    _ema_backbone_state,
    sha256_file,
)


def test_block_causal_mask_allows_current_and_previous_blocks():
    mask = block_causal_attention_mask(6, 2)
    assert mask[0, 1]
    assert mask[0, 2]
    assert not mask[2, 0]
    assert mask[2, 3]
    assert mask[2, 4]


def test_two_stream_mask_exposes_only_clean_prefix_and_noisy_current_block():
    mask = two_stream_attention_mask(5, 2)
    noisy_query = 5 + 3
    assert not mask[noisy_query, 0]
    assert mask[noisy_query, 3]
    assert not mask[noisy_query, 5 + 4]
    assert mask[noisy_query, 5 + 1]


@pytest.mark.parametrize("layer0_long_residual_scale", [0.0, 1.0])
def test_two_stream_current_logits_cannot_read_clean_current_or_future(
    layer0_long_residual_scale,
):
    config = MarlinDecoderConfig(
        vocab_size=7,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=6,
        block_width=2,
        fingerprint_bits=8,
        dropout=0.0,
        layer0_long_residual_scale=layer0_long_residual_scale,
        mask_token_id=4,
        pad_token_id=0,
    )
    decoder = MarlinDecoder(config).eval()
    clean = torch.tensor([[1, 5, 6, 5, 6, 2]])
    changed = torch.tensor([[1, 5, 6, 6, 5, 3]])
    noised = torch.tensor([[1, 5, 6, 4, 4, 4]])
    mass = torch.tensor([100.0])
    fingerprint = torch.zeros((1, 8))

    first = decoder.two_stream_logits(clean, noised, mass, fingerprint)
    second = decoder.two_stream_logits(changed, noised, mass, fingerprint)

    assert torch.allclose(first[:, 3:5], second[:, 3:5], atol=1e-6)


def test_sampling_logits_use_revealed_tokens_inside_current_block():
    config = MarlinDecoderConfig(
        vocab_size=9,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=8,
        block_width=4,
        fingerprint_bits=8,
        dropout=0.0,
        mask_token_id=4,
        pad_token_id=0,
    )
    decoder = MarlinDecoder(config).eval()
    all_masked = torch.tensor([[1, 4, 4, 4, 4]])
    partially_revealed = torch.tensor([[1, 5, 4, 4, 4]])
    mass = torch.tensor([100.0])
    fingerprint = torch.zeros((1, 8))

    masked_logits = decoder.sampling_logits(all_masked, mass, fingerprint)
    revealed_logits = decoder.sampling_logits(
        partially_revealed,
        mass,
        fingerprint,
    )

    assert not torch.allclose(masked_logits[:, 2:], revealed_logits[:, 2:])


def test_symmetric_noise_preserves_number_of_on_bits():
    fingerprint = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.float32)
    generator = torch.Generator().manual_seed(7)
    noisy = symmetric_fingerprint_noise(
        fingerprint,
        corruption_probability=1.0,
        min_fraction=0.5,
        max_fraction=0.5,
        generator=generator,
    )
    assert noisy.sum() == fingerprint.sum()
    assert not torch.equal(noisy, fingerprint)


def test_conditioner_emits_mass_isotope_and_active_bit_tokens():
    conditioner = MarlinConditioner(
        hidden_size=28, fingerprint_bits=8, num_mass_frequencies=4
    )
    fingerprint = torch.tensor([[1, 0, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)
    tokens, mask = conditioner(
        torch.tensor([250.0]), fingerprint, torch.tensor([[0.1, 0.02]])
    )
    assert tokens.shape == (1, 4, 28)
    assert mask.tolist() == [[True, True, True, True]]


def test_conditioner_omits_disabled_isotope_token():
    conditioner = MarlinConditioner(
        hidden_size=28, fingerprint_bits=8, num_mass_frequencies=4
    )
    fingerprint = torch.tensor([[1, 0, 1, 0, 0, 0, 0, 0]], dtype=torch.float32)

    tokens, mask = conditioner(torch.tensor([250.0]), fingerprint)

    assert tokens.shape == (1, 3, 28)
    assert mask.tolist() == [[True, True, True]]


def test_frigid_compatible_decoder_propagates_layer_norm_epsilon():
    backbone_epsilon = 3e-7
    cross_attention_epsilon = 4e-7
    fingerprint_epsilon = 5e-7
    model = MarlinDecoder(
        MarlinDecoderConfig(
            vocab_size=8,
            hidden_size=8,
            num_layers=2,
            num_heads=1,
            intermediate_size=16,
            max_length=5,
            block_width=2,
            fingerprint_bits=8,
            dropout=0.0,
            fingerprint_self_attention_layers=2,
            frigid_compatible_layer_order=True,
            layer_norm_eps=backbone_epsilon,
            cross_attention_layer_norm_eps=cross_attention_epsilon,
            fingerprint_layer_norm_eps=fingerprint_epsilon,
        )
    )

    backbone_layer_norms = [
        model.embedding_norm,
        model.prediction_norm,
        *(
            norm
            for layer in model.layers
            for norm in (layer.norm1, layer.norm3)
        ),
    ]
    cross_attention_layer_norms = [layer.norm2 for layer in model.layers]
    fingerprint_layer_norms = [
        model.conditioner.fingerprint.layer_norm,
        *(layer.norm for layer in model.conditioner.fingerprint.self_attention_layers),
    ]

    assert all(norm.eps == backbone_epsilon for norm in backbone_layer_norms)
    assert all(
        norm.eps == cross_attention_epsilon
        for norm in cross_attention_layer_norms
    )
    assert all(norm.eps == fingerprint_epsilon for norm in fingerprint_layer_norms)


def test_marlin_decoder_defaults_to_frigid_layer_norm_epsilon():
    assert MarlinDecoderConfig().layer_norm_eps == 1e-12
    assert MarlinDecoderConfig().cross_attention_layer_norm_eps == 1e-5
    assert MarlinDecoderConfig().fingerprint_layer_norm_eps == 1e-5
    assert MarlinDecoderConfig().layer0_long_residual_scale == 0.0


@pytest.mark.parametrize(
    "scale",
    [-1.0, float("nan"), float("inf"), -float("inf")],
)
def test_marlin_decoder_rejects_invalid_layer0_long_residual_scale(scale):
    with pytest.raises(
        ValueError,
        match="layer0_long_residual_scale must be finite and non-negative",
    ):
        MarlinDecoderConfig(layer0_long_residual_scale=scale)


def test_layer0_long_residual_is_exact_and_state_dict_compatible():
    common = dict(
        vocab_size=9,
        hidden_size=8,
        num_layers=2,
        num_heads=1,
        intermediate_size=16,
        max_length=6,
        block_width=2,
        fingerprint_bits=8,
        dropout=0.0,
        mask_token_id=4,
        pad_token_id=0,
    )
    baseline = MarlinDecoder(
        MarlinDecoderConfig(**common, layer0_long_residual_scale=0.0)
    ).eval()
    residual = MarlinDecoder(
        MarlinDecoderConfig(**common, layer0_long_residual_scale=1.0)
    ).eval()
    residual.load_state_dict(baseline.state_dict(), strict=True)
    assert residual.state_dict().keys() == baseline.state_dict().keys()

    input_ids = torch.tensor([[1, 5, 4, 4, 4, 4]])
    mass = torch.tensor([100.0])
    fingerprint = torch.zeros((1, 8))

    def capture_head_inputs(model):
        captured = {}

        def capture_layer0(_module, _inputs, output):
            captured["layer0"] = output.detach().clone()

        def capture_pre_head(_module, inputs):
            captured["pre_head"] = inputs[0].detach().clone()

        handles = [
            model.layers[0].register_forward_hook(capture_layer0),
            model.prediction_dense.register_forward_pre_hook(capture_pre_head),
        ]
        try:
            with torch.inference_mode():
                model(input_ids, mass, fingerprint)
        finally:
            for handle in handles:
                handle.remove()
        return captured

    baseline_inputs = capture_head_inputs(baseline)
    residual_inputs = capture_head_inputs(residual)

    assert torch.equal(residual_inputs["layer0"], baseline_inputs["layer0"])
    assert torch.allclose(
        residual_inputs["pre_head"],
        baseline_inputs["pre_head"] + residual_inputs["layer0"],
        atol=1e-6,
    )


def test_marlin_uses_fixed_decay_ema():
    config = MarlinDecoderConfig(
        vocab_size=8,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=5,
        block_width=2,
        fingerprint_bits=4,
        dropout=0.0,
        mask_token_id=3,
        pad_token_id=0,
    )
    module = MarlinLightningModule(config, ema_decay=0.9999)

    assert module.ema.decay == 0.9999
    assert module.ema.num_updates is None


def test_molecular_validation_callback_builds_bounded_oracle_set(tmp_path):
    metadata = tmp_path / "validation.csv"
    metadata.write_text("smiles\nCCO\ninvalid\n")
    tokenizer = load_safe_tokenizer(
        "/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/data/safe-gpt/tokenizer.json"
    )

    callback = MarlinMolecularValidationCallback(
        tokenizer,
        metadata,
        output_dir=tmp_path,
        fingerprint_bits=16,
        every_n_steps=10,
        samples=1,
        candidates=2,
    )

    assert len(callback.records) == 1
    assert callback.records[0]["smiles"] == "CCO"
    assert callback.records[0]["fingerprint"].shape == (16,)
    assert callback.output_path == tmp_path / "molecular_validation.jsonl"
    assert (
        callback.latest_output_path
        == tmp_path / "molecular_validation_latest.json"
    )


def test_molecular_validation_can_evaluate_raw_weights_without_touching_ema(
    tmp_path,
):
    class FailingEma:
        def store(self, _parameters):
            raise AssertionError("raw validation must not store EMA parameters")

        def copy_to(self, _parameters):
            raise AssertionError("raw validation must not copy EMA parameters")

        def restore(self, _parameters):
            raise AssertionError("raw validation must not restore EMA parameters")

    class EmptySampler:
        def generate_ranked_with_stats(self, *args, **kwargs):
            del args, kwargs
            return [], SimpleNamespace(
                attempts=2,
                valid=0,
                strict_valid=0,
                mass_valid=0,
                unique_mass_valid=0,
                constraint_dead_ends=2,
                eos_terminated=0,
                max_length_terminated=0,
                sample_terminal_safes=[],
                sample_dead_ends=[{"safe": "broken", "heavy_mass": 12.0}],
            )

    metadata = tmp_path / "validation.csv"
    metadata.write_text("smiles\nCCO\n")
    tokenizer = load_safe_tokenizer(
        "/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/data/safe-gpt/tokenizer.json"
    )
    callback = MarlinMolecularValidationCallback(
        tokenizer,
        metadata,
        output_dir=tmp_path,
        fingerprint_bits=16,
        every_n_steps=10,
        samples=1,
        candidates=2,
        use_ema=False,
    )
    callback._sampler = lambda _model: EmptySampler()
    config = MarlinDecoderConfig(
        vocab_size=8,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=5,
        block_width=2,
        fingerprint_bits=4,
        dropout=0.0,
        mask_token_id=3,
        pad_token_id=0,
    )
    module = MarlinLightningModule(config)
    module.ema = FailingEma()
    module.log = lambda *args, **kwargs: None
    trainer = SimpleNamespace(
        global_step=10,
        is_global_zero=True,
        current_epoch=0,
    )

    callback.on_train_batch_end(trainer, module, None, None, 0)

    result = json.loads(callback.latest_output_path.read_text())
    assert result["weights"] == "raw"
    assert result["metrics"]["validity"] == 0.0
    assert result["metrics"]["strict_validity"] == 0.0
    assert result["metrics"]["constraint_dead_end_rate"] == 1.0
    assert result["samples"][0]["dead_end_examples"][0]["safe"] == "broken"


def test_molecular_validation_callback_reports_clearml_table_and_image(tmp_path):
    class RecordingLogger:
        def __init__(self):
            self.tables = []
            self.images = []

        def report_table(self, **kwargs):
            self.tables.append(kwargs)

        def report_image(self, **kwargs):
            self.images.append(kwargs)

    class RecordingTask:
        def __init__(self):
            self.logger = RecordingLogger()

        def get_logger(self):
            return self.logger

    metadata = tmp_path / "validation.csv"
    metadata.write_text("smiles\nCCO\n")
    tokenizer = load_safe_tokenizer(
        "/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/data/safe-gpt/tokenizer.json"
    )
    task = RecordingTask()
    callback = MarlinMolecularValidationCallback(
        tokenizer,
        metadata,
        output_dir=tmp_path,
        fingerprint_bits=16,
        every_n_steps=10,
        samples=1,
        candidates=2,
        clearml_task=task,
    )

    callback._report_clearml_samples(
        step=10,
        epoch=2.0,
        sample_rows=[
            {
                "target_smiles": "CCO",
                "top1_smiles": "CCO",
                "generated_smiles": "CCO",
                "tanimoto": 1.0,
                "mass_error_ppm": 0.0,
                "valid": 2,
                "mass_valid": 2,
            }
        ],
    )

    assert len(task.logger.tables) == 1
    assert len(task.logger.images) == 1
    assert task.logger.tables[0]["iteration"] == 10
    assert task.logger.images[0]["max_image_history"] == -1


def test_theoretical_isotope_ratios_include_m_plus_one_and_two():
    ratios = theoretical_isotope_ratios(Chem.MolFromSmiles("CCl"))
    assert ratios.shape == (2,)
    assert ratios[0] > 0
    assert ratios[1] > 0.3


def test_mass_shell_prunes_overshoot_and_forbids_early_eos():
    constraint = MassShellConstraint(
        [0.0, 12.0, 16.0],
        [0, 1, 1],
        [0.0, 4.0, 2.0],
        eos_token_id=0,
        ppm_tolerance=10,
    )
    logits = constraint.apply(torch.zeros(3), MassShellState(heavy_mass=90.0), 100.0)
    assert torch.isneginf(logits[0])
    assert torch.isneginf(logits[1])
    assert torch.isneginf(logits[2])


def test_mass_shell_boosts_eos_when_no_nonzero_token_fits():
    constraint = MassShellConstraint(
        [0.0, 12.0, 16.0],
        [0, 1, 1],
        [0.0, 4.0, 2.0],
        eos_token_id=0,
        ppm_tolerance=10,
        eos_boost=1.5,
    )
    logits = constraint.apply(
        torch.zeros(3),
        MassShellState(heavy_mass=95.0, heavy_atoms=1, valence_sum=4.0),
        100.0,
    )
    assert logits[0] == 1.5
    assert torch.isneginf(logits[1:]).all()


def test_token_properties_ignore_safe_grammar_characters():
    properties = token_properties("C1=CC(Cl)=CC=C1")
    assert properties.heavy_atoms == 7
    assert properties.heavy_mass > 100


def test_token_properties_cover_rare_official_safe_elements():
    properties = token_properties("[208Tl+]")
    assert properties.heavy_atoms == 1
    assert properties.heavy_mass > 200


def test_token_properties_use_explicit_isotope_mass():
    boron_10 = token_properties("[10B]")
    boron_default = token_properties("[B]")
    carbon_13 = token_properties("[13C]")
    carbon_default = token_properties("[C]")

    assert boron_10.heavy_atoms == 1
    assert boron_10.heavy_mass < boron_default.heavy_mass
    assert carbon_13.heavy_atoms == 1
    assert carbon_13.heavy_mass > carbon_default.heavy_mass


def test_token_properties_ignore_unknown_safe_bracket_tokens():
    properties = token_properties("[Z]")
    assert properties.heavy_atoms == 0
    assert properties.heavy_mass == 0.0
    assert properties.valence_sum == 0.0


def test_official_safe_vocabulary_builds_token_property_table():
    tokenizer = load_safe_tokenizer(
        "/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/data/safe-gpt/tokenizer.json"
    )
    special_ids = {
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.mask_token_id,
        tokenizer.pad_token_id,
    }
    masses, atom_counts, valences = build_token_property_table(
        len(tokenizer), tokenizer.convert_ids_to_tokens, special_ids
    )

    assert len(masses) == len(tokenizer)
    assert len(atom_counts) == len(tokenizer)
    assert len(valences) == len(tokenizer)


def test_training_collator_refuses_to_truncate_safe(monkeypatch):
    class OverlengthTokenizer:
        def __call__(self, values, **kwargs):
            assert values == ["C"]
            assert kwargs["truncation"] is False
            return {"input_ids": torch.tensor([[1, 4, 5, 2]])}

    monkeypatch.setattr("marlin.training.safe_to_smiles", lambda safe, fix: safe)
    collator = MarlinCollator(
        OverlengthTokenizer(), max_length=3, fingerprint_bits=16
    )

    with pytest.raises(ValueError, match="refusing silent truncation"):
        collator([{"safe": "C"}])


def test_stream_filter_excludes_invalid_overlength_and_test_safe(
    monkeypatch, tmp_path
):
    class LengthTokenizer:
        def encode(self, safe, add_special_tokens):
            assert add_special_tokens is True
            return list(range(len(safe) + 2))

    exclusions = tmp_path / "exclude.csv"
    exclusions.write_text("inchikey\nOKKJLVBELUTLKV-UHFFFAOYSA-N\n")
    monkeypatch.setattr(
        "marlin.training.safe_to_smiles",
        lambda safe, fix: {"CC": "CC", "CO": "CO"}.get(safe),
    )
    stream_filter = MarlinTrainingFilter(LengthTokenizer(), 4, exclusions)

    assert stream_filter({"safe": "CC"})
    assert not stream_filter({"safe": "CO"})
    assert not stream_filter({"safe": "CCC"})
    assert not stream_filter({"safe": "invalid"})
    assert not stream_filter({})


def test_small_decoder_forward_and_loss():
    config = MarlinDecoderConfig(
        vocab_size=32,
        hidden_size=28,
        num_layers=2,
        num_heads=4,
        intermediate_size=56,
        max_length=8,
        block_width=2,
        fingerprint_bits=16,
        dropout=0.0,
        mask_token_id=3,
        pad_token_id=0,
    )
    model = MarlinDecoder(config)
    tokens = torch.tensor([[1, 5, 6, 2, 0, 0]])
    mass = torch.tensor([100.0])
    fingerprint = torch.zeros((1, 16))
    fingerprint[0, [2, 7]] = 1
    logits = model(tokens, mass, fingerprint)
    sampling_logits = model.sampling_logits(tokens, mass, fingerprint)
    expected_sampling_logits = model.two_stream_logits(
        tokens, tokens, mass, fingerprint
    )
    loss = model.diffusion_loss(
        tokens, mass, fingerprint, generator=torch.Generator().manual_seed(3)
    )
    assert logits.shape == (1, 6, 32)
    assert torch.equal(sampling_logits, expected_sampling_logits)
    assert torch.isfinite(loss)


def test_diffusion_objective_reports_reconstruction_metrics():
    config = MarlinDecoderConfig(
        vocab_size=8,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=5,
        block_width=2,
        fingerprint_bits=4,
        dropout=0.0,
        mask_token_id=3,
        pad_token_id=0,
    )
    model = MarlinDecoder(config)
    loss, metrics = model.diffusion_objective(
        torch.tensor([[1, 4, 5, 6, 2]]),
        torch.tensor([50.0]),
        torch.zeros((1, 4)),
        generator=torch.Generator().manual_seed(7),
        collect_metrics=True,
    )

    assert torch.isfinite(loss)
    assert set(metrics) == {
        "masked_token_accuracy_top1",
        "masked_token_accuracy_top10",
        "masked_target_probability",
        "masked_token_nll",
        "masked_token_perplexity",
        "masked_prediction_entropy",
        "masked_top1_confidence",
        "masked_argmax_eos_fraction",
        "masked_eos_target_count",
        "masked_eos_target_probability",
        "masked_eos_target_rank",
        "mask_fraction",
        "full_sequence_mask_fraction",
        "masked_sequence_accuracy",
    }
    assert all(torch.isfinite(value) for value in metrics.values())
    assert 0 <= metrics["masked_token_accuracy_top1"] <= 1
    assert 0 <= metrics["masked_token_accuracy_top10"] <= 1
    assert metrics["masked_token_accuracy_top10"] >= metrics["masked_token_accuracy_top1"]
    assert 0 <= metrics["masked_top1_confidence"] <= 1
    assert 0 <= metrics["masked_argmax_eos_fraction"] <= 1
    assert metrics["masked_eos_target_count"] >= 0


def test_diffusion_keeps_bos_clean():
    config = MarlinDecoderConfig(
        vocab_size=8,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=5,
        block_width=2,
        fingerprint_bits=4,
        dropout=0.0,
        mask_token_id=3,
        pad_token_id=0,
    )
    model = MarlinDecoder(config)
    tokens = torch.tensor([[1, 4, 5, 6, 2]])
    captured = {}
    original = model.two_stream_logits

    def capture(
        clean_ids,
        noised_ids,
        precursor_mass,
        fingerprint,
        isotope_ratios=None,
        **kwargs,
    ):
        captured["noised_ids"] = noised_ids.clone()
        captured["isotope_ratios"] = isotope_ratios
        return original(
            clean_ids,
            noised_ids,
            precursor_mass,
            fingerprint,
            isotope_ratios,
            **kwargs,
        )

    model.two_stream_logits = capture
    model.diffusion_loss(
        tokens,
        torch.tensor([50.0]),
        torch.zeros((1, 4)),
        generator=torch.Generator().manual_seed(1),
    )

    assert captured["noised_ids"][0, 0] == tokens[0, 0]


def test_diffusion_passes_isotope_ratios_to_conditioner():
    config = MarlinDecoderConfig(
        vocab_size=8,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=5,
        block_width=2,
        fingerprint_bits=4,
        dropout=0.0,
        mask_token_id=3,
        pad_token_id=0,
    )
    model = MarlinDecoder(config)
    captured = {}
    original = model.conditioner.forward

    def capture(precursor_mass, fingerprint, isotope_ratios=None, **kwargs):
        captured["isotope_ratios"] = isotope_ratios
        return original(precursor_mass, fingerprint, isotope_ratios, **kwargs)

    model.conditioner.forward = capture
    isotope_ratios = torch.tensor([[0.12, 0.03]])
    model.diffusion_loss(
        torch.tensor([[1, 4, 5, 6, 2]]),
        torch.tensor([50.0]),
        torch.zeros((1, 4)),
        isotope_ratios=isotope_ratios,
        generator=torch.Generator().manual_seed(1),
    )
    assert torch.equal(captured["isotope_ratios"], isotope_ratios)


def test_diffusion_loss_averages_the_weighted_token_sum_over_blocks():
    config = MarlinDecoderConfig(
        vocab_size=8,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=5,
        block_width=2,
        fingerprint_bits=4,
        dropout=0.0,
        mask_token_id=3,
        pad_token_id=0,
    )
    model = MarlinDecoder(config)
    for parameter in model.parameters():
        parameter.data.zero_()
    tokens = torch.tensor([[1, 4, 5, 6, 2]])
    seed = 5
    expected_generator = torch.Generator().manual_seed(seed)
    times = torch.rand((1, 2), generator=expected_generator).clamp_min(1e-4)
    probabilities = times[:, torch.tensor([0, 0, 0, 1, 1])]
    valid = torch.tensor([[False, True, True, True, True]])
    masked = (
        torch.rand(tokens.shape, generator=expected_generator) < probabilities
    ) & valid
    expected = (
        masked.float().mul(probabilities.reciprocal()).sum()
        * torch.log(torch.tensor(float(config.vocab_size)))
        / 2
    )

    loss = model.diffusion_loss(
        tokens,
        torch.tensor([50.0]),
        torch.zeros((1, 4)),
        generator=torch.Generator().manual_seed(seed),
    )

    assert torch.allclose(loss, expected)


def test_balanced_token_loss_does_not_reweight_eos_targets():
    config = MarlinDecoderConfig(
        vocab_size=8,
        hidden_size=8,
        num_layers=1,
        num_heads=1,
        intermediate_size=16,
        max_length=5,
        block_width=2,
        fingerprint_bits=4,
        dropout=0.0,
        eos_token_id=2,
        mask_token_id=3,
        pad_token_id=0,
    )
    model = MarlinDecoder(config)
    for parameter in model.parameters():
        parameter.data.zero_()
    tokens = torch.tensor([[1, 2, 5, 5, 2]])
    seed = 9
    expected_generator = torch.Generator().manual_seed(seed)
    times = torch.rand((1, 2), generator=expected_generator).clamp_min(1e-4)
    probabilities = times[:, torch.tensor([0, 0, 0, 1, 1])]
    valid = torch.tensor([[False, True, True, True, True]])
    masked = (
        torch.rand(tokens.shape, generator=expected_generator) < probabilities
    ) & valid
    eos_targets = valid & tokens.eq(config.eos_token_id)
    masked = masked | (
        torch.rand(tokens.shape, generator=expected_generator) < 1.0
    ) & eos_targets
    target_weights = torch.ones_like(probabilities)
    target_weights = target_weights.masked_fill(eos_targets, 7.0)
    expected = (
        (target_weights * masked * probabilities.reciprocal()).sum()
        * torch.log(torch.tensor(float(config.vocab_size)))
        / 2
    )

    loss = model.diffusion_loss(
        tokens,
        torch.tensor([50.0]),
        torch.zeros((1, 4)),
        generator=torch.Generator().manual_seed(seed),
        eos_loss_weight=7.0,
        eos_mask_probability=1.0,
        balanced_token_loss_alpha=1.0,
        token_loss_weight_max=20.0,
    )

    assert torch.allclose(loss, expected)


def test_attention_warm_start_concatenates_qkv():
    attention = torch.nn.MultiheadAttention(4, 1, batch_first=True)
    state = {}
    for offset, part in enumerate(("query", "key", "value")):
        state[f"x.{part}.weight"] = torch.full((4, 4), float(offset + 1))
        state[f"x.{part}.bias"] = torch.full((4,), float(offset + 1))
    _copy_attention(attention, state, "x", "test")
    assert torch.equal(attention.in_proj_weight[:4], state["x.query.weight"])
    assert torch.equal(attention.in_proj_weight[-4:], state["x.value.weight"])


def test_frigid_warm_start_loads_complete_embedding_stack():
    model = MarlinDecoder(
        MarlinDecoderConfig(
            vocab_size=8,
            hidden_size=4,
            num_layers=1,
            num_heads=1,
            intermediate_size=4,
            max_length=4,
            fingerprint_bits=8,
            dropout=0.0,
            frigid_compatible_layer_order=True,
        )
    )
    prefix = "backbone.bert.embeddings"
    state = {
        f"{prefix}.word_embeddings.weight": torch.full((8, 4), 1.0),
        f"{prefix}.position_embeddings.weight": torch.full((4, 4), 2.0),
        f"{prefix}.token_type_embeddings.weight": torch.full((2, 4), 3.0),
        f"{prefix}.LayerNorm.weight": torch.full((4,), 4.0),
        f"{prefix}.LayerNorm.bias": torch.full((4,), 5.0),
    }

    _copy_embeddings(model, state, prefix)

    assert torch.equal(
        model.token_type_embedding.weight,
        state[f"{prefix}.token_type_embeddings.weight"],
    )
    assert torch.equal(
        model.embedding_norm.bias,
        state[f"{prefix}.LayerNorm.bias"],
    )


def test_frigid_warm_start_maps_ema_to_unique_backbone_parameters():
    tied = torch.tensor([3.0])
    raw_fingerprint = torch.tensor([5.0])
    checkpoint = {
        "state_dict": {
            "backbone.embedding.weight": tied,
            "backbone.layer.weight": torch.tensor([4.0]),
            "backbone.decoder.weight": tied,
            "fingerprint.embedding.weight": raw_fingerprint,
        },
        "ema": {
            "shadow_params": [torch.tensor([30.0]), torch.tensor([40.0])],
        },
    }

    effective = _ema_backbone_state(checkpoint)

    assert torch.equal(
        effective["backbone.embedding.weight"], torch.tensor([30.0])
    )
    assert torch.equal(effective["backbone.layer.weight"], torch.tensor([40.0]))
    assert effective["backbone.decoder.weight"] is effective[
        "backbone.embedding.weight"
    ]
    assert effective["fingerprint.embedding.weight"] is raw_fingerprint


def test_fingerprint_warm_start_loads_set_encoder():
    encoder = SparseFingerprintEncoder(
        bits=8,
        hidden_size=4,
        num_heads=1,
        num_self_attention_layers=1,
        dropout=0.0,
    )
    prefix = "fingerprint_conditioner.fingerprint_encoder"
    state = {
        f"{prefix}.bit_embeddings.weight": torch.full((8, 4), 1.0),
        f"{prefix}.layer_norm.weight": torch.full((4,), 2.0),
        f"{prefix}.layer_norm.bias": torch.full((4,), 3.0),
    }
    for offset, part in enumerate(("query", "key", "value"), start=4):
        state[f"{prefix}.self_attention_layers.0.{part}.weight"] = torch.full(
            (4, 4), float(offset)
        )
        state[f"{prefix}.self_attention_layers.0.{part}.bias"] = torch.full(
            (4,), float(offset)
        )
    state[f"{prefix}.self_attention_layers.0.output_dense.weight"] = torch.full(
        (4, 4), 7.0
    )
    state[f"{prefix}.self_attention_layers.0.output_dense.bias"] = torch.full(
        (4,), 8.0
    )
    state[f"{prefix}.self_attention_layers.0.output_layer_norm.weight"] = (
        torch.full((4,), 9.0)
    )
    state[f"{prefix}.self_attention_layers.0.output_layer_norm.bias"] = (
        torch.full((4,), 10.0)
    )

    _copy_fingerprint_encoder(encoder, state, prefix)

    assert torch.equal(encoder.embedding.weight, state[f"{prefix}.bit_embeddings.weight"])
    assert torch.equal(encoder.layer_norm.weight, state[f"{prefix}.layer_norm.weight"])
    layer = encoder.self_attention_layers[0]
    assert torch.equal(
        layer.attention.in_proj_weight[:4],
        state[f"{prefix}.self_attention_layers.0.query.weight"],
    )
    assert torch.equal(
        layer.norm.bias,
        state[f"{prefix}.self_attention_layers.0.output_layer_norm.bias"],
    )


def test_frigid_compatible_layer_applies_ffn_before_cross_attention():
    events = []
    config = MarlinDecoderConfig(
        vocab_size=8,
        hidden_size=4,
        num_layers=1,
        num_heads=1,
        intermediate_size=4,
        max_length=4,
        block_width=2,
        fingerprint_bits=8,
        dropout=0.0,
        frigid_compatible_layer_order=True,
    )
    layer = MarlinDecoder(config).layers[0]

    class AttentionRecorder(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def forward(self, query, key, value, **kwargs):
            events.append(self.name)
            return torch.zeros_like(query), None

    class Recorder(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def forward(self, value):
            events.append(self.name)
            return value

    layer.self_attention = AttentionRecorder("self_attention")
    layer.cross_attention = AttentionRecorder("cross_attention")
    layer.norm1 = Recorder("norm1")
    layer.linear1 = Recorder("linear1")
    layer.linear2 = Recorder("linear2")
    layer.norm2 = Recorder("norm2")
    layer.norm3 = Recorder("norm3")
    layer(
        torch.zeros((1, 2, 4)),
        torch.zeros((1, 1, 4)),
        attention_mask=torch.zeros((2, 2), dtype=torch.bool),
        padding_mask=None,
        condition_padding_mask=None,
    )

    assert events == [
        "self_attention",
        "norm1",
        "linear1",
        "linear2",
        "norm3",
        "cross_attention",
        "norm2",
    ]


def test_warm_start_sha256_file(tmp_path):
    checkpoint = tmp_path / "DLM.ckpt"
    checkpoint.write_bytes(b"official checkpoint bytes")

    assert (
        sha256_file(checkpoint)
        == "39508f80120dcafa9564e61482326a83ebe2b06d09d2bee7b3ee0e1fbc932d97"
    )


def test_batched_sampler_reports_attempt_validity_and_uniqueness():
    class FixedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = MarlinDecoderConfig(
                vocab_size=4,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=3,
                block_width=2,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            logits = torch.zeros((*input_ids.shape, 4), device=input_ids.device)
            logits[..., 1] = 10.0
            return logits

    molecule = Chem.MolFromSmiles("C")
    target_mass = Descriptors.ExactMolWt(molecule)
    constraint = MassShellConstraint(
        [0.0] * 4,
        eos_token_id=2,
        ppm_tolerance=10,
    )
    sampler = MarlinSampler(
        FixedModel(),
        constraint,
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda _: "C",
        safe_to_smiles=lambda _: "C",
        strict_safe_to_smiles=lambda _: None,
        forbidden_token_ids=(0, 3),
    )
    ranked, stats = sampler.generate_ranked_with_stats(
        torch.zeros(8), target_mass, candidates=3
    )
    assert stats.attempts == 3
    assert stats.valid == 3
    assert stats.strict_valid == 0
    assert stats.mass_valid == 3
    assert stats.unique_mass_valid == 1
    assert len(ranked) == 1


def test_generate_one_uses_model_confidence_reveal_order():
    class PositionConfidenceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.seen = []
            self.config = MarlinDecoderConfig(
                vocab_size=5,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=3,
                block_width=2,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            self.seen.append(input_ids.detach().clone())
            logits = torch.full((*input_ids.shape, 5), -torch.inf)
            logits[:, 1, 1] = 2.0
            logits[:, 1, 4] = 1.0
            logits[:, 2, 1] = 5.0
            logits[:, 2, 4] = 0.0
            return logits

    model = PositionConfidenceModel()
    target_mass = Descriptors.ExactMolWt(Chem.MolFromSmiles("CC"))
    sampler = MarlinSampler(
        model,
        MassShellConstraint(
            [0.0, 12.0, 0.0, 0.0, 16.0],
            [0, 1, 0, 0, 1],
            [0.0, 4.0, 0.0, 0.0, 2.0],
            eos_token_id=2,
            ppm_tolerance=10,
        ),
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "".join({1: "C", 4: "O"}.get(i, "") for i in ids),
        safe_to_smiles=lambda safe: safe,
        grammar_mask=lambda _prefix, logits, _mass: logits,
        forbidden_token_ids=(0, 3),
    )

    result = sampler.generate_one(torch.zeros(8), target_mass)

    assert result == ("CC", "CC")
    assert model.seen[1].tolist() == [[0, 3, 1]]


def test_constrained_sampler_uses_model_confidence_reveal_order():
    class SuffixConfidentModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.seen = []
            self.config = MarlinDecoderConfig(
                vocab_size=5,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=3,
                block_width=2,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            self.seen.append(input_ids.detach().clone())
            logits = torch.full(
                (*input_ids.shape, 5),
                -torch.inf,
                device=input_ids.device,
            )
            logits[:, 1, 1] = 1.0
            logits[:, 1, 4] = 0.0
            logits[:, 2, 1] = 10.0
            logits[:, 2, 4] = 0.0
            return logits

    model = SuffixConfidentModel()
    sampler = MarlinSampler(
        model,
        MassShellConstraint(
            [0.0, 12.0, 0.0, 0.0, 16.0],
            [0, 1, 0, 0, 1],
            [0.0, 4.0, 0.0, 0.0, 2.0],
            eos_token_id=2,
            ppm_tolerance=10,
        ),
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "".join(
            {1: "C", 4: "O"}.get(token_id, "") for token_id in ids
        ),
        safe_to_smiles=lambda safe: safe,
        grammar_mask=lambda _prefix, logits, _mass: logits,
        forbidden_token_ids=(0, 3),
    )

    sampler.generate_ranked_with_stats(
        torch.zeros(8),
        target_mass=Descriptors.ExactMolWt(Chem.MolFromSmiles("CC")),
        candidates=1,
        generator=torch.Generator().manual_seed(7),
    )

    assert model.seen[1].tolist() == [[0, 3, 1]]


def test_constrained_sampler_does_not_reveal_eos_before_an_earlier_block_hole():
    class EarlyEosModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.seen = []
            self.config = MarlinDecoderConfig(
                vocab_size=7,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=3,
                block_width=2,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=4,
                pad_token_id=3,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            self.seen.append(input_ids.detach().clone())
            logits = torch.full(
                (*input_ids.shape, 7),
                -torch.inf,
                device=input_ids.device,
            )
            if input_ids[0, 2].item() == 2:
                logits[:, 1, 5] = 10.0
            else:
                logits[:, 1, 5] = 0.0
                logits[:, 1, 6] = 0.0
            logits[:, 2, 2] = 10.0
            logits[:, 2, 6] = 8.0
            return logits

    tokens = ("[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "C", "O")
    grammar = SafeGrammarMask(
        tokens,
        lambda ids: "".join(tokens[index] for index in ids if index >= 5),
        eos_token_id=2,
        mask_token_id=4,
        special_token_ids=(0, 1, 2, 3, 4),
    )
    model = EarlyEosModel()
    target_mass = Descriptors.ExactMolWt(Chem.MolFromSmiles("C"))
    sampler = MarlinSampler(
        model,
        MassShellConstraint(
            [0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 15.99491462],
            [0, 0, 0, 0, 0, 1, 1],
            [0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 2.0],
            eos_token_id=2,
            ppm_tolerance=10,
        ),
        bos_token_id=1,
        eos_token_id=2,
        mask_token_id=4,
        decode_tokens=lambda ids: "".join(
            tokens[index] for index in ids if index >= 5
        ),
        safe_to_smiles=lambda safe: safe or None,
        grammar_mask=grammar,
        forbidden_token_ids=(0, 1, 3, 4),
    )

    ranked, stats = sampler.generate_ranked_with_stats(
        torch.zeros(8),
        target_mass,
        candidates=1,
        generator=torch.Generator().manual_seed(7),
    )

    assert model.seen[1].tolist() == [[1, 5, 4]]
    assert stats.mass_valid == 1
    assert [candidate.smiles for candidate in ranked] == ["C"]


def test_constrained_sampler_reports_confidence_order_dead_end():
    class SuffixBiasedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.seen = []
            self.config = MarlinDecoderConfig(
                vocab_size=5,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=3,
                block_width=2,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            self.seen.append(input_ids.detach().clone())
            logits = torch.full((*input_ids.shape, 5), -torch.inf, device=input_ids.device)
            logits[:, 1, 1] = 1.0
            logits[:, 1, 4] = 0.0
            if input_ids[0, 1].item() == 1:
                logits[:, 2, 2] = 10.0
            else:
                logits[:, 2, 4] = 10.0
            return logits

    model = SuffixBiasedModel()
    target_mass = Descriptors.ExactMolWt(Chem.MolFromSmiles("C"))
    sampler = MarlinSampler(
        model,
        MassShellConstraint(
            [0.0, 12.0, 0.0, 0.0, 16.0],
            [0, 1, 0, 0, 1],
            [0.0, 4.0, 0.0, 0.0, 2.0],
            eos_token_id=2,
            ppm_tolerance=10,
        ),
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "C" if 1 in ids else "",
        safe_to_smiles=lambda safe: safe or None,
        grammar_mask=lambda _prefix, logits, _mass: logits,
        forbidden_token_ids=(0, 3),
    )

    assert sampler.generate_one(torch.zeros(8), target_mass) is None
    assert model.seen[1].tolist() == [[0, 3, 4]]


def test_sampler_can_disable_mass_shell_for_unconstrained_diagnostics():
    class SuffixConfidentModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.seen = []
            self.config = MarlinDecoderConfig(
                vocab_size=4,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=3,
                block_width=2,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            self.seen.append(input_ids.detach().clone())
            logits = torch.full((*input_ids.shape, 4), -torch.inf, device=input_ids.device)
            logits[:, 1, 2] = 1.0
            logits[:, 1, 1] = 0.0
            logits[:, 2, 1] = 10.0
            logits[:, 2, 2] = 0.0
            return logits

    model = SuffixConfidentModel()
    sampler = MarlinSampler(
        model,
        MassShellConstraint(
            [0.0, 12.0, 0.0, 0.0],
            [0, 1, 0, 0],
            [0.0, 4.0, 0.0, 0.0],
            eos_token_id=2,
            ppm_tolerance=10,
        ),
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "C" if 1 in ids else "",
        safe_to_smiles=lambda safe: safe or None,
        forbidden_token_ids=(0, 3),
        mass_shell_enabled=False,
    )

    sampler.generate_ranked_with_stats(torch.zeros(8), target_mass=12.0, candidates=1)

    assert model.seen[1].tolist() == [[0, 3, 1]]


def test_batched_sampler_returns_valid_candidates_when_mass_shell_is_disabled():
    class CarbonModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = MarlinDecoderConfig(
                vocab_size=4,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=2,
                block_width=1,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            logits = torch.full((*input_ids.shape, 4), -torch.inf, device=input_ids.device)
            logits[..., 1] = 0.0
            return logits

    sampler = MarlinSampler(
        CarbonModel(),
        MassShellConstraint(
            [0.0, 12.0, 0.0, 0.0],
            [0, 1, 0, 0],
            [0.0, 4.0, 0.0, 0.0],
            eos_token_id=2,
            ppm_tolerance=10,
        ),
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "C" * ids.count(1),
        safe_to_smiles=lambda safe: safe,
        forbidden_token_ids=(0, 3),
        mass_shell_enabled=False,
    )

    ranked, stats = sampler.generate_ranked_with_stats(
        torch.zeros(8), target_mass=100.0, candidates=1
    )

    assert stats.valid == 1
    assert stats.mass_valid == 0
    assert stats.unique_mass_valid == 0
    assert [candidate.smiles for candidate in ranked] == ["C"]


@pytest.mark.parametrize("generation_mode", ["block", "canvas"])
def test_batched_sampler_uses_argmax_token_deterministically(
    monkeypatch, generation_mode
):
    class AmbiguousModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = MarlinDecoderConfig(
                vocab_size=5,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=3,
                block_width=1,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            logits = torch.full(
                (*input_ids.shape, 5),
                -torch.inf,
                device=input_ids.device,
            )
            logits[..., 1] = 2.0
            logits[..., 4] = 1.0
            return logits

    def fail_if_sampled(*_args, **_kwargs):
        pytest.fail("token selection must use argmax, not multinomial sampling")

    monkeypatch.setattr(torch, "multinomial", fail_if_sampled)
    sampler = MarlinSampler(
        AmbiguousModel(),
        MassShellConstraint(
            [0.0, 12.0, 0.0, 0.0, 16.0],
            [0, 1, 0, 0, 1],
            [0.0, 4.0, 0.0, 0.0, 2.0],
            eos_token_id=2,
            ppm_tolerance=10,
        ),
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "".join(
            {1: "C", 4: "O"}.get(token_id, "") for token_id in ids
        ),
        safe_to_smiles=lambda safe: safe or None,
        forbidden_token_ids=(0, 3),
        mass_shell_enabled=False,
        generation_mode=generation_mode,
    )

    ranked, stats = sampler.generate_ranked_with_stats(
        torch.zeros(8),
        target_mass=12.0,
        candidates=1,
        diversity_dropout=0.0,
    )

    assert stats.valid == 1
    assert [candidate.smiles for candidate in ranked] == ["C"]


def test_constrained_beam_recovers_mass_valid_second_choice():
    class GreedyTrapModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = MarlinDecoderConfig(
                vocab_size=5,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=3,
                block_width=2,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            del precursor_mass, fingerprint
            logits = torch.full((*input_ids.shape, 5), -torch.inf, device=input_ids.device)
            for row in range(input_ids.shape[0]):
                if input_ids[row, 1].item() == 3:
                    logits[row, 1, 4] = 2.0
                    logits[row, 1, 1] = 1.0
                else:
                    logits[row, 2, 2] = 5.0
            return logits

    tokens = ("[UNK]", "C", "[SEP]", "[MASK]", "O")
    target_mass = Descriptors.ExactMolWt(Chem.MolFromSmiles("C"))
    sampler = MarlinSampler(
        GreedyTrapModel(),
        MassShellConstraint(
            [0.0, 12.0, 0.0, 0.0, 15.99491462],
            [0, 1, 0, 0, 1],
            [0.0, 4.0, 0.0, 0.0, 2.0],
            eos_token_id=2,
            ppm_tolerance=10,
        ),
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "".join(
            tokens[index] for index in ids if index in {1, 4}
        ),
        safe_to_smiles=lambda safe: safe or None,
        forbidden_token_ids=(0, 3),
    )

    greedy, _ = sampler.generate_ranked_with_stats(
        torch.zeros(8),
        target_mass,
        candidates=1,
        diversity_dropout=0.0,
    )
    beam, stats = sampler.generate_beam_ranked_with_stats(
        torch.zeros(8),
        target_mass,
        beam_width=2,
        branch_factor=2,
    )

    assert greedy == []
    assert [candidate.smiles for candidate in beam] == ["C"]
    assert stats.completed_paths == 1
    assert stats.mass_valid_paths == 1
    assert stats.expanded_hypotheses == 3
    assert stats.eos_terminated == 2
    assert stats.backtrack_recoveries == 1
    assert stats.max_committed_tokens == 1


def test_constrained_beam_bounds_completed_pool():
    class TwoTerminalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = MarlinDecoderConfig(
                vocab_size=5,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=2,
                block_width=1,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            del precursor_mass, fingerprint
            logits = torch.full(
                (*input_ids.shape, 5), -torch.inf, device=input_ids.device
            )
            logits[:, 1, 1] = 2.0
            logits[:, 1, 4] = 1.0
            return logits

    sampler = MarlinSampler(
        TwoTerminalModel(),
        MassShellConstraint(
            [0.0, 12.0, 0.0, 0.0, 14.0],
            [0, 1, 0, 0, 1],
            [0.0, 4.0, 0.0, 0.0, 3.0],
            eos_token_id=2,
            ppm_tolerance=10,
        ),
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "".join(
            {1: "C", 4: "N"}.get(token_id, "") for token_id in ids
        ),
        safe_to_smiles=lambda safe: safe or None,
        forbidden_token_ids=(0, 3),
        mass_shell_enabled=False,
    )

    ranked, stats = sampler.generate_beam_ranked_with_stats(
        torch.zeros(8),
        target_mass=100.0,
        beam_width=1,
        branch_factor=2,
    )

    assert [candidate.smiles for candidate in ranked] == ["C"]
    assert stats.completed_paths == 1
    assert stats.expanded_tokens == 2
    assert stats.block_terminated == 2
    assert stats.pruned_hypotheses == 1


def test_constrained_beam_considers_accepted_eos_outside_branch_factor():
    class LowRankEosModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = MarlinDecoderConfig(
                vocab_size=4,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=3,
                block_width=2,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            del precursor_mass, fingerprint
            logits = torch.full(
                (*input_ids.shape, 4), -torch.inf, device=input_ids.device
            )
            for row in range(input_ids.shape[0]):
                position = int((input_ids[row] == 3).nonzero()[0])
                logits[row, position, 1] = 3.0
                if position == 1:
                    logits[row, position, 2] = 4.0
                else:
                    logits[row, position, 2] = 0.0
            return logits

    sampler = MarlinSampler(
        LowRankEosModel(),
        MassShellConstraint(
            [0.0, 12.0, 0.0, 0.0],
            [0, 1, 0, 0],
            [0.0, 4.0, 0.0, 0.0],
            eos_token_id=2,
            ppm_tolerance=10,
        ),
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "C" * ids.count(1),
        safe_to_smiles=lambda safe: safe or None,
        forbidden_token_ids=(0, 3),
        mass_shell_enabled=False,
    )

    target_mass = Descriptors.ExactMolWt(Chem.MolFromSmiles("C"))
    ranked, stats = sampler.generate_beam_ranked_with_stats(
        torch.zeros(8),
        target_mass,
        beam_width=2,
        branch_factor=1,
    )

    assert [candidate.smiles for candidate in ranked] == ["C", "CC"]
    assert stats.completed_paths == 2
    assert stats.eos_terminated == 2
    assert stats.independent_eos_probes == 2
    assert stats.expanded_tokens == 4


def test_constrained_beam_invalid_eos_does_not_exhaust_live_frontier():
    class InvalidEosTrapModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.batch_sizes = []
            self.config = MarlinDecoderConfig(
                vocab_size=5,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=3,
                block_width=2,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            del precursor_mass, fingerprint
            self.batch_sizes.append(input_ids.shape[0])
            logits = torch.full(
                (*input_ids.shape, 5), -torch.inf, device=input_ids.device
            )
            for row in range(input_ids.shape[0]):
                if input_ids[row, 1].item() == 3:
                    logits[row, 1, 2] = 3.0
                    logits[row, 1, 1] = 2.0
                    logits[row, 1, 4] = 1.0
                else:
                    logits[row, 2, 1] = 3.0
            return logits

    model = InvalidEosTrapModel()
    sampler = MarlinSampler(
        model,
        MassShellConstraint(
            [0.0, 12.0, 0.0, 0.0, 16.0],
            [0, 1, 0, 0, 1],
            [0.0, 4.0, 0.0, 0.0, 2.0],
            eos_token_id=2,
            ppm_tolerance=10,
        ),
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "".join(
            {1: "C", 4: "O"}.get(token_id, "") for token_id in ids
        ),
        safe_to_smiles=lambda safe: safe or None,
        forbidden_token_ids=(0, 3),
        mass_shell_enabled=False,
    )

    ranked, stats = sampler.generate_beam_ranked_with_stats(
        torch.zeros(8),
        target_mass=100.0,
        beam_width=2,
        branch_factor=3,
        max_model_batch_size=1,
    )

    assert {candidate.smiles for candidate in ranked} == {"CC", "CO"}
    assert stats.completed_paths == 2
    assert stats.eos_terminated == 1
    assert stats.backtrack_recoveries == 2
    assert stats.max_committed_tokens == 2
    assert max(model.batch_sizes) == 1


def test_constrained_beam_handles_partial_final_block():
    class PartialBlockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.canvases = []
            self.config = MarlinDecoderConfig(
                vocab_size=4,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=4,
                block_width=2,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            del precursor_mass, fingerprint
            self.canvases.extend(input_ids.detach().cpu().tolist())
            logits = torch.full(
                (*input_ids.shape, 4), -torch.inf, device=input_ids.device
            )
            for row in range(input_ids.shape[0]):
                position = input_ids[row].tolist().index(3)
                logits[row, position, 1] = 1.0
            return logits

    model = PartialBlockModel()
    sampler = MarlinSampler(
        model,
        MassShellConstraint(
            [0.0, 12.0, 0.0, 0.0],
            [0, 1, 0, 0],
            [0.0, 4.0, 0.0, 0.0],
            eos_token_id=2,
            ppm_tolerance=10,
        ),
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "".join("C" for token_id in ids if token_id == 1),
        safe_to_smiles=lambda safe: safe if len(safe) >= 3 else None,
        forbidden_token_ids=(0, 2, 3),
        mass_shell_enabled=False,
    )

    ranked, stats = sampler.generate_beam_ranked_with_stats(
        torch.zeros(8),
        target_mass=100.0,
        beam_width=1,
        branch_factor=1,
    )

    assert [candidate.smiles for candidate in ranked] == ["CCC"]
    assert model.canvases == [
        [0, 3, 3],
        [0, 1, 3],
        [0, 1, 1, 3],
    ]
    assert max(len(canvas) for canvas in model.canvases) == 4
    assert stats.max_committed_tokens == 3


def test_constrained_beam_eos_ignores_masked_suffix():
    class EosModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.canvases = []
            self.config = MarlinDecoderConfig(
                vocab_size=4,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=4,
                block_width=3,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            del precursor_mass, fingerprint
            self.canvases.extend(input_ids.detach().cpu().tolist())
            logits = torch.full(
                (*input_ids.shape, 4), -torch.inf, device=input_ids.device
            )
            for row in range(input_ids.shape[0]):
                position = input_ids[row].tolist().index(3)
                logits[row, position, 1 if position == 1 else 2] = 1.0
            return logits

    model = EosModel()
    sampler = MarlinSampler(
        model,
        MassShellConstraint(
            [0.0, 12.0, 0.0, 0.0],
            [0, 1, 0, 0],
            [0.0, 4.0, 0.0, 0.0],
            eos_token_id=2,
            ppm_tolerance=10,
        ),
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "".join("C" for token_id in ids if token_id == 1),
        safe_to_smiles=lambda safe: safe or None,
        forbidden_token_ids=(0, 3),
        mass_shell_enabled=False,
    )

    ranked, stats = sampler.generate_beam_ranked_with_stats(
        torch.zeros(8),
        target_mass=100.0,
        beam_width=1,
        branch_factor=1,
    )

    assert [candidate.smiles for candidate in ranked] == ["C"]
    assert model.canvases == [[0, 3, 3, 3], [0, 1, 3, 3]]
    assert stats.eos_terminated == 1
    assert stats.max_committed_tokens == 1
    assert stats.completion_paths[0]["content_tokens"] == 1


@pytest.mark.parametrize(
    ("target_mass", "kwargs", "message"),
    [
        (100.0, {"temperature": float("nan")}, "temperature"),
        (100.0, {"temperature": float("inf")}, "temperature"),
        (0.0, {}, "target_mass"),
        (-1.0, {}, "target_mass"),
    ],
)
def test_constrained_beam_rejects_nonfinite_or_nonpositive_controls(
    target_mass,
    kwargs,
    message,
):
    class MinimalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = MarlinDecoderConfig(
                vocab_size=4,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=2,
                block_width=1,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

    sampler = MarlinSampler(
        MinimalModel(),
        MassShellConstraint(
            [0.0, 12.0, 0.0, 0.0],
            [0, 1, 0, 0],
            [0.0, 4.0, 0.0, 0.0],
            eos_token_id=2,
            ppm_tolerance=10,
        ),
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "",
        safe_to_smiles=lambda safe: None,
    )

    with pytest.raises(ValueError, match=message):
        sampler.generate_beam_ranked_with_stats(
            torch.zeros(8),
            target_mass=target_mass,
            **kwargs,
        )


def test_canvas_sampler_fills_fixed_masked_sequence():
    class CanvasModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.seen = []
            self.config = MarlinDecoderConfig(
                vocab_size=5,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=4,
                block_width=1,
                fingerprint_bits=8,
                dropout=0.0,
                eos_token_id=2,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            self.seen.append(input_ids.detach().clone())
            logits = torch.full((*input_ids.shape, 5), -torch.inf, device=input_ids.device)
            logits[..., 1] = 10.0
            logits[..., 2] = 9.0
            logits[..., 3] = 8.0
            return logits

    sampler = MarlinSampler(
        CanvasModel(),
        MassShellConstraint(
            [0.0, 12.0, 0.0, 0.0, 16.0],
            [0, 1, 0, 0, 1],
            [0.0, 4.0, 0.0, 0.0, 2.0],
            eos_token_id=2,
            ppm_tolerance=10,
        ),
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "C" * ids.count(1),
        safe_to_smiles=lambda safe: safe,
        forbidden_token_ids=(0, 3),
        mass_shell_enabled=False,
        generation_mode="canvas",
    )

    ranked, stats = sampler.generate_ranked_with_stats(
        torch.zeros(8),
        target_mass=12.0,
        candidates=1,
        generator=torch.Generator().manual_seed(7),
    )

    assert stats.valid == 1
    assert ranked[0].smiles in {"C", "CC"}
    first_seen = sampler.model.seen[0].tolist()[0]
    assert first_seen[0] == 0
    assert first_seen[-1] == 2
    assert first_seen[1:-1] in ([3], [3, 3])


def test_batched_sampler_discards_tokens_after_eos_before_mass_and_decoding():
    class EosThenJunkModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = MarlinDecoderConfig(
                vocab_size=5,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=4,
                block_width=3,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            logits = torch.zeros((*input_ids.shape, 5), device=input_ids.device)
            logits[:, 1, 1] = 20.0
            if input_ids.shape[1] > 2:
                logits[:, 2, 2] = 15.0
                logits[:, 3, 4] = 10.0
            return logits

    target_mass = Descriptors.ExactMolWt(Chem.MolFromSmiles("C"))
    constraint = MassShellConstraint(
        [0.0, 12.0, 0.0, 0.0, 16.0],
        [0, 1, 0, 0, 1],
        [0.0, 4.0, 0.0, 0.0, 2.0],
        eos_token_id=2,
        ppm_tolerance=10,
    )
    sampler = MarlinSampler(
        EosThenJunkModel(),
        constraint,
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: (
            "".join({1: "C", 4: "X"}.get(i, "") for i in ids) if 2 in ids else ""
        ),
        safe_to_smiles=lambda safe: safe if safe == "C" else None,
        forbidden_token_ids=(0, 3),
    )

    assert sampler.generate_one(torch.zeros(8), target_mass) == ("C", "C")
    ranked, stats = sampler.generate_ranked_with_stats(
        torch.zeros(8), target_mass, candidates=1
    )

    assert stats.valid == 1
    assert stats.mass_valid == 1
    assert [candidate.smiles for candidate in ranked] == ["C"]


def test_batched_sampler_charges_suffix_before_rejecting_dead_end():
    class SuffixFirstModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.seen = []
            self.config = MarlinDecoderConfig(
                vocab_size=7,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=4,
                block_width=3,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            self.seen.append(input_ids.detach().clone())
            logits = torch.full((*input_ids.shape, 7), -torch.inf)
            logits[:, 1, 1] = 10.0
            logits[:, 1, 5] = 0.0
            logits[:, 2, 2] = 5.0
            logits[:, 2, 5] = 0.0
            logits[:, 2, 6] = 0.0
            logits[:, 3, 4] = 20.0
            logits[:, 3, 5] = 0.0
            return logits

    model = SuffixFirstModel()
    target_mass = Descriptors.ExactMolWt(Chem.MolFromSmiles("C"))
    sampler = MarlinSampler(
        model,
        MassShellConstraint(
            [0.0, 12.0, 0.0, 0.0, 16.0, 0.0, 0.0],
            [0, 1, 0, 0, 1, 0, 0],
            [0.0, 4.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            eos_token_id=2,
            ppm_tolerance=10,
        ),
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "C" if 1 in ids and 2 in ids else "",
        safe_to_smiles=lambda safe: safe or None,
        forbidden_token_ids=(0, 3),
    )

    assert sampler.generate_one(torch.zeros(8), target_mass) is None
    assert model.seen[1].tolist() == [[0, 3, 3, 4]]
    model.seen.clear()
    ranked, stats = sampler.generate_ranked_with_stats(
        torch.zeros(8), target_mass, candidates=1
    )

    assert model.seen[1].tolist() == [[0, 3, 3, 4]]
    assert stats.eos_terminated == 1
    assert stats.mass_valid == 0
    assert ranked == []


def test_mass_state_counts_committed_tokens_across_holes_and_stops_at_eos():
    class MinimalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = MarlinDecoderConfig(vocab_size=6, mask_token_id=3)

    constraint = MassShellConstraint(
        [0.0, 12.0, 0.0, 0.0, 16.0, 14.0],
        [0, 1, 0, 0, 1, 1],
        [0.0, 4.0, 0.0, 0.0, 2.0, 3.0],
        eos_token_id=2,
    )
    sampler = MarlinSampler(
        MinimalModel(),
        constraint,
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda _: "",
        safe_to_smiles=lambda _: None,
    )

    state = sampler._mass_state([0, 3, 1, 3, 4, 2, 5])

    assert state == MassShellState(
        heavy_mass=28.0,
        heavy_atoms=2,
        valence_sum=6.0,
    )


def test_batched_sampler_keeps_mass_valid_candidate_at_max_length():
    class CarbonModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = MarlinDecoderConfig(
                vocab_size=4,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=3,
                block_width=1,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            logits = torch.full(
                (*input_ids.shape, 4), -torch.inf, device=input_ids.device
            )
            logits[..., 1] = 0.0
            return logits

    target_mass = Descriptors.ExactMolWt(Chem.MolFromSmiles("CC"))
    constraint = MassShellConstraint(
        [0.0, 12.0, 0.0, 0.0],
        [0, 1, 0, 0],
        [0.0, 4.0, 0.0, 0.0],
        eos_token_id=2,
        ppm_tolerance=10,
    )
    sampler = MarlinSampler(
        CarbonModel(),
        constraint,
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "C" * ids.count(1),
        safe_to_smiles=lambda safe: safe,
        forbidden_token_ids=(0, 3),
    )

    ranked, stats = sampler.generate_ranked_with_stats(
        torch.zeros(8), target_mass, candidates=1
    )

    assert stats.valid == 1
    assert stats.mass_valid == 1
    assert [candidate.smiles for candidate in ranked] == ["CC"]


def test_batched_sampler_aligns_first_generated_block_after_bos():
    class BlockAlignedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.config = MarlinDecoderConfig(
                vocab_size=4,
                hidden_size=4,
                num_layers=1,
                num_heads=1,
                intermediate_size=4,
                max_length=4,
                block_width=2,
                fingerprint_bits=8,
                dropout=0.0,
                mask_token_id=3,
                pad_token_id=0,
            )

        def forward(self, input_ids, precursor_mass, fingerprint):
            assert input_ids.shape[1] == 3
            logits = torch.full(
                (*input_ids.shape, 4), -torch.inf, device=input_ids.device
            )
            logits[:, 1, 1] = 1.0
            logits[:, 2, 2] = 0.0
            return logits

    target_mass = Descriptors.ExactMolWt(Chem.MolFromSmiles("C"))
    sampler = MarlinSampler(
        BlockAlignedModel(),
        MassShellConstraint(
            [0.0, 12.0, 0.0, 0.0],
            [0, 1, 0, 0],
            [0.0, 4.0, 0.0, 0.0],
            eos_token_id=2,
            ppm_tolerance=10,
        ),
        bos_token_id=0,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "C" if 1 in ids else "",
        safe_to_smiles=lambda safe: safe,
        forbidden_token_ids=(0, 3),
    )

    ranked, stats = sampler.generate_ranked_with_stats(
        torch.zeros(8), target_mass, candidates=1
    )

    assert stats.mass_valid == 1
    assert [candidate.smiles for candidate in ranked] == ["C"]
