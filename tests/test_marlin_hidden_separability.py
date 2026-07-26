import pytest
import torch

from marlin.hidden_separability import (
    ALL_STAGE_NAMES,
    capture_action_stages,
    identical_prefix_conflict_groups,
    normalized_rms_per_row,
    summarize_conflict_top1,
    summarize_intervention,
    symmetric_normalized_rms,
)
from marlin.model import MarlinDecoder, MarlinDecoderConfig


def tiny_decoder() -> MarlinDecoder:
    torch.manual_seed(7)
    return MarlinDecoder(
        MarlinDecoderConfig(
            vocab_size=9,
            hidden_size=8,
            num_layers=2,
            num_heads=1,
            intermediate_size=16,
            max_length=8,
            block_width=4,
            fingerprint_bits=8,
            dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            mask_token_id=3,
            pad_token_id=0,
        )
    ).eval()


def test_capture_action_stages_selects_noisy_stream_positions_in_fp32():
    model = tiny_decoder()
    inputs = torch.tensor(
        [
            [1, 4, 3, 3, 3],
            [1, 5, 6, 3, 3],
        ]
    )
    masses = torch.tensor([100.0, 100.0])
    fingerprints = torch.tensor(
        [
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )

    captured = capture_action_stages(
        model,
        inputs,
        masses,
        fingerprints,
        torch.tensor([2, 3]),
    )

    assert set(captured) == set(ALL_STAGE_NAMES)
    assert captured["raw_logits"].shape == (2, 9)
    for stage in ALL_STAGE_NAMES[:-1]:
        assert captured[stage].shape == (2, 8)
    assert all(value.dtype == torch.float32 for value in captured.values())
    assert all(value.device.type == "cpu" for value in captured.values())
    assert not model.layers[0].self_attention._forward_hooks
    assert not model.layers[0].cross_attention._forward_hooks
    assert not model.layers[-1]._forward_hooks
    assert not model.prediction_norm._forward_hooks

    full_prediction_norm = {}

    def save_prediction_norm(_module, _inputs, output):
        full_prediction_norm["value"] = output.detach().clone()

    handle = model.prediction_norm.register_forward_hook(
        save_prediction_norm
    )
    expected_logits = model.sampling_logits(inputs, masses, fingerprints)
    handle.remove()
    batch_indices = torch.arange(2)
    positions = torch.tensor([2, 3])
    assert torch.allclose(
        captured["prediction_norm_post_head"],
        full_prediction_norm["value"][batch_indices, positions + inputs.shape[1]],
    )
    assert torch.allclose(
        captured["raw_logits"],
        expected_logits[batch_indices, positions],
    )

    with pytest.raises(ValueError, match="integer dtype"):
        capture_action_stages(
            model,
            inputs,
            masses,
            fingerprints,
            torch.tensor([2.0, 3.0]),
        )


def test_normalized_rms_and_intervention_summary_are_paired_by_row():
    correct = {
        stage: torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        for stage in ALL_STAGE_NAMES[:-1]
    }
    correct["raw_logits"] = torch.tensor(
        [
            [0.0, 4.0, 2.0],
            [0.0, 1.0, 3.0],
        ]
    )
    intervention = {
        stage: value.clone() for stage, value in correct.items()
    }
    for stage in ALL_STAGE_NAMES[:-1]:
        intervention[stage][0] = 0.0
    intervention["raw_logits"] = torch.tensor(
        [
            [0.0, 1.0, 3.0],
            [0.0, 1.0, 3.0],
        ]
    )

    delta = normalized_rms_per_row(
        correct["first_self_attention_output"],
        intervention["first_self_attention_output"],
    )
    summary = summarize_intervention(
        correct,
        intervention,
        torch.tensor([1, 2]),
        c_token_id=2,
    )

    assert delta.tolist() == pytest.approx([1.0, 0.0])
    assert summary["normalized_rms"]["first_self_attention_output"][
        "median"
    ] == pytest.approx(0.5)
    assert summary["normalized_rms"]["first_self_attention_output"][
        "p90"
    ] == pytest.approx(0.9)
    assert summary["raw_argmax_change_count"] == 1
    assert summary["raw_argmax_change_rate"] == pytest.approx(0.5)
    assert summary["correct_raw_top1_accuracy"] == 1.0
    assert summary["intervention_raw_top1_accuracy"] == 0.5
    assert summary["target_vs_c_logit_margin"]["non_c_targets"][
        "correct"
    ]["median"] == pytest.approx(2.0)


def test_identical_prefix_conflicts_require_distinct_targets():
    actions = [
        {"position": 2, "canvas_ids": (1, 4, 3), "target_id": 5},
        {"position": 2, "canvas_ids": (1, 4, 3), "target_id": 6},
        {"position": 2, "canvas_ids": (1, 7, 3), "target_id": 5},
        {"position": 3, "canvas_ids": (1, 4, 5, 3), "target_id": 8},
        {"position": 3, "canvas_ids": (1, 4, 5, 3), "target_id": 8},
    ]

    conflicts = identical_prefix_conflict_groups(actions)

    assert conflicts == [
        {
            "group_id": "position-2-conflict-0",
            "position": 2,
            "prefix_ids": [1, 4],
            "action_indices": [0, 1],
            "target_ids": [5, 6],
        }
    ]
    assert symmetric_normalized_rms(
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
    ) == pytest.approx(1.4142135)


def test_conflict_top1_summary_reports_restricted_and_global_accuracy():
    summary = summarize_conflict_top1(
        target_ids=[5, 6, 7, 5],
        restricted_top1_ids=[5, 6, 5, 5],
        global_top1_ids=[5, 5, 5, 5],
    )

    assert summary == {
        "rows": 4,
        "restricted_candidate_top1_correct": 3,
        "restricted_candidate_top1_accuracy": 0.75,
        "global_raw_top1_correct": 2,
        "global_raw_top1_accuracy": 0.5,
    }
