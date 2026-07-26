import pytest
import torch

from marlin.mass_shell import MassShellConstraint
from marlin.model import MarlinDecoderConfig
from marlin.prefix_diagnostic import (
    build_production_prefix_actions,
    diagnose_production_prefix,
    summarize_prefix_rows,
)
from marlin.sampler import MarlinSampler


TOKENS = ("[PAD]", "[BOS]", "[EOS]", "[MASK]", "C", "O", "N")


class PrefixActionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.seen = []
        self.config = MarlinDecoderConfig(
            vocab_size=len(TOKENS),
            hidden_size=4,
            num_layers=1,
            num_heads=1,
            intermediate_size=4,
            max_length=5,
            block_width=2,
            fingerprint_bits=8,
            dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            mask_token_id=3,
            pad_token_id=0,
        )

    def sampling_logits(self, input_ids, precursor_mass, fingerprint):
        self.seen.append(input_ids.detach().clone())
        logits = torch.zeros(
            (*input_ids.shape, len(TOKENS)),
            device=input_ids.device,
        )
        logits[..., 3] = 10.0
        logits[:, 1, 4] = 8.0
        if input_ids.shape[1] > 2:
            logits[:, 2, 5] = 8.0
        if input_ids.shape[1] > 3:
            logits[:, 3, 2] = 8.0
        return logits


def make_sampler(*, grammar_mask=None):
    model = PrefixActionModel()
    constraint = MassShellConstraint(
        [0.0, 0.0, 0.0, 0.0, 12.0, 15.99491462, 14.003074004],
        [0, 0, 0, 0, 1, 1, 1],
        [0.0, 0.0, 0.0, 0.0, 4.0, 2.0, 3.0],
        eos_token_id=2,
        ppm_tolerance=10.0,
    )
    return model, MarlinSampler(
        model,
        constraint,
        bos_token_id=1,
        eos_token_id=2,
        mask_token_id=3,
        decode_tokens=lambda ids: "".join(
            TOKENS[token_id] for token_id in ids if token_id >= 4
        ),
        safe_to_smiles=lambda safe: safe or None,
        grammar_mask=grammar_mask,
        forbidden_token_ids=(0, 1, 3),
    )


def test_prefix_diagnostic_stops_before_redundant_eos_block():
    model, sampler = make_sampler()

    result = diagnose_production_prefix(
        sampler,
        [1, 4, 5, 2],
        torch.zeros(8),
        32.026214748,
        lambda token_id: TOKENS[token_id],
    )

    assert [canvas.tolist() for canvas in model.seen] == [
        [[1, 3, 3]],
        [[1, 4, 3]],
    ]
    assert [action["target_token"] for action in result["actions"]] == [
        "C",
        "O",
    ]
    assert [action["raw"]["top1_id"] for action in result["actions"]] == [3, 3]
    assert [
        action["constrained"]["top1_id"] for action in result["actions"]
    ] == [4, 5]
    assert result["summary"]["target_allowed_rate"] == 1.0
    assert result["summary"]["constrained_top1_accuracy"] == 1.0
    assert result["summary"]["raw_top1_accuracy"] == 0.0


def test_prefix_diagnostic_still_scores_eos_inside_current_block():
    model, sampler = make_sampler()

    result = diagnose_production_prefix(
        sampler,
        [1, 4, 2],
        torch.zeros(8),
        16.031300128,
        lambda token_id: TOKENS[token_id],
    )

    assert [canvas.tolist() for canvas in model.seen] == [
        [[1, 3, 3]],
        [[1, 4, 3]],
    ]
    assert [action["target_token"] for action in result["actions"]] == [
        "C",
        "[EOS]",
    ]


def test_production_prefix_action_builder_snapshots_canvas_before_reveal():
    _, sampler = make_sampler()

    actions = build_production_prefix_actions(
        sampler,
        [1, 4, 5, 2],
        32.026214748,
    )

    assert actions == [
        {
            "position": 1,
            "block_index": 0,
            "block_offset": 0,
            "canvas_length": 3,
            "canvas_ids": (1, 3, 3),
            "target_id": 4,
        },
        {
            "position": 2,
            "block_index": 0,
            "block_offset": 1,
            "canvas_length": 3,
            "canvas_ids": (1, 4, 3),
            "target_id": 5,
        },
    ]


def test_prefix_diagnostic_records_disallowed_target_and_continues_to_eos():
    def reject_carbon(prefix_ids, logits, _target_mass):
        constrained = logits.clone()
        if prefix_ids == [1]:
            constrained[4] = -torch.inf
        return constrained

    model, sampler = make_sampler(grammar_mask=reject_carbon)

    result = diagnose_production_prefix(
        sampler,
        [1, 4, 2],
        torch.zeros(8),
        16.031300128,
        lambda token_id: TOKENS[token_id],
    )

    assert len(model.seen) == 2
    assert [action["target_allowed"] for action in result["actions"]] == [
        False,
        True,
    ]
    assert result["actions"][0]["constrained"]["target_rank"] is None
    assert result["actions"][0]["constrained"]["target_probability"] == 0.0
    assert result["summary"]["target_allowed"] == 1
    assert result["summary"]["first_disallowed_position"] == 1


@pytest.mark.parametrize("target_id", (0, 1, 3, 6))
def test_prefix_diagnostic_rejects_forbidden_target_token(target_id):
    _, sampler = make_sampler()
    # ID 6 stands in for the tokenizer's UNK ID in this tiny vocabulary.
    sampler.forbidden_token_ids = (*sampler.forbidden_token_ids, 6)

    with pytest.raises(
        ValueError,
        match=rf"forbidden target token ID {target_id} at position 2",
    ):
        diagnose_production_prefix(
            sampler,
            [1, 4, target_id, 2],
            torch.zeros(8),
            32.026214748,
            lambda token_id: TOKENS[token_id],
        )


def test_prefix_summary_reports_row_aware_first_disallowed_locator():
    def reject_oxygen(prefix_ids, logits, _target_mass):
        constrained = logits.clone()
        if prefix_ids == [1, 4]:
            constrained[5] = -torch.inf
        return constrained

    _, allowed_sampler = make_sampler()
    allowed = diagnose_production_prefix(
        allowed_sampler,
        [1, 4, 2],
        torch.zeros(8),
        16.031300128,
        lambda token_id: TOKENS[token_id],
    )
    _, rejected_sampler = make_sampler(grammar_mask=reject_oxygen)
    rejected = diagnose_production_prefix(
        rejected_sampler,
        [1, 4, 5, 2],
        torch.zeros(8),
        32.026214748,
        lambda token_id: TOKENS[token_id],
    )
    summary = summarize_prefix_rows(
        [
            {"metadata_row": 3, "actions": allowed["actions"]},
            {"metadata_row": 17, "actions": rejected["actions"]},
        ]
    )

    assert summary["first_disallowed"] == {
        "metadata_row": 17,
        "position": 2,
    }
    assert summary["first_disallowed_position"] == 2
