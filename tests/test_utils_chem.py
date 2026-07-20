"""Regression tests for strict SAFE decoding."""

from dlm.utils.utils_chem import safe_to_smiles


def test_safe_to_smiles_does_not_repair_unmatched_attachment_when_strict():
    assert safe_to_smiles("C%10", fix=False) is None
    assert safe_to_smiles("C%10", fix=True) == "C"
