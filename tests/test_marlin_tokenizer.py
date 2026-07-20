import pytest

from marlin.tokenizer import validate_safe_tokenizer


class FakeTokenizer:
    unk_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 3
    mask_token_id = 4

    def __len__(self):
        return 1880


def test_validate_safe_tokenizer_accepts_exact_contract():
    actual = validate_safe_tokenizer(
        FakeTokenizer(),
        expected_vocab_size=1880,
        expected_special_token_ids={
            "unk": 0,
            "bos": 1,
            "eos": 2,
            "pad": 3,
            "mask": 4,
        },
    )

    assert actual["mask"] == 4


def test_validate_safe_tokenizer_rejects_vocabulary_mismatch():
    with pytest.raises(ValueError, match="1880 tokens; expected 1879"):
        validate_safe_tokenizer(
            FakeTokenizer(),
            expected_vocab_size=1879,
            expected_special_token_ids={
                "unk": 0,
                "bos": 1,
                "eos": 2,
                "pad": 3,
                "mask": 4,
            },
        )
