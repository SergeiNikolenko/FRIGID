import torch

from marlin.grammar import (
    SafeGrammarMask,
    _has_reachable_exact_mass,
    _has_mass_viable_atom_completion,
    _has_mass_viable_bracket_completion,
    _has_mass_viable_percent_completion,
    _scan,
)


def test_safe_grammar_accepts_balanced_ring_and_rejects_same_atom_closure():
    assert _scan("C1CCCCC1").terminal
    assert _scan("C11") is None
    assert _scan("C1CCCCC12CCCCC2").terminal
    assert _scan("O1O1(") is None
    assert _scan("C(C)(C)(C)(C)(") is None


def test_safe_grammar_requires_balanced_terminal_structure():
    assert not _scan("C1CC").terminal
    assert not _scan("C(C").terminal
    assert _scan("C(C)O").terminal
    assert _scan("CCC").minimum_mass(0) > 36.0


def test_safe_grammar_accepts_branch_bonds_and_partial_vocabulary_tokens():
    assert _scan("C(=O)O").terminal
    assert _scan("[NH") is not None
    assert not _scan("[NH").terminal
    assert _scan("[NH+]").terminal
    assert _scan("C[N+](C)(C)C").terminal
    assert _scan("[13C@@H]").terminal
    assert _scan("[2H]") is None
    assert _scan("[671") is not None
    assert _scan("[6711") is None
    assert _scan("[671Y") is None
    assert _scan("[13C-11") is not None
    assert _scan("[13C-111") is None
    assert _scan("[NHHE") is None
    assert _scan("c[nH]c").terminal
    assert _scan("C1[2H]1") is None
    assert _scan("C%1") is not None
    assert not _scan("C%1").terminal
    assert _scan("C1CCCCC-1").terminal


def test_safe_grammar_accepts_real_nplib1_target():
    target = "COc1cc(C2C3(O)C(O)C4CC2(O)C(O)(C(=O)O4)C3C(=O)c2ccccc2)oc(=O)c1"
    assert _scan(target).terminal


def test_safe_grammar_masks_invalid_highest_scoring_token():
    tokens = ("[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "C", "1")
    grammar = SafeGrammarMask(
        tokens,
        lambda ids: "".join(tokens[index] for index in ids if index >= 5),
        eos_token_id=2,
        special_token_ids=(0, 1, 2, 3, 4),
    )
    logits = torch.full((len(tokens),), -10.0)
    logits[6] = 10.0
    logits[5] = 9.0

    constrained = grammar([], logits)

    assert torch.isneginf(constrained[6])
    assert int(constrained.argmax()) == 5


def test_safe_grammar_rejects_bracket_when_no_element_fits_mass_shell():
    target_mass = 444.103807533379
    assert not _has_mass_viable_bracket_completion(
        "C" * 32 + "[", target_mass, 4.0, target_mass * 10e-6
    )


def test_safe_grammar_rejects_incomplete_syntax_when_no_atom_fits_mass_shell():
    target_mass = 444.103807533379
    tolerance = target_mass * 10e-6

    assert not _has_mass_viable_atom_completion(
        "C" * 32 + "(", target_mass, 4.0, tolerance
    )
    assert not _has_mass_viable_percent_completion(
        "C" * 32 + "1(=%3", target_mass, 4.0, tolerance
    )


def test_safe_grammar_rejects_ring_opening_when_no_later_atom_fits_mass_shell():
    tokens = ("[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "1")
    prefix = "C" * 32
    grammar = SafeGrammarMask(
        tokens,
        lambda _ids: prefix,
        eos_token_id=2,
        special_token_ids=(0, 1, 2, 3, 4),
    )
    logits = torch.full((len(tokens),), -torch.inf)
    logits[5] = 1.0

    constrained = grammar([], logits, 444.103807533379)

    assert torch.isneginf(constrained).all()


def test_safe_grammar_prunes_unreachable_exact_mass_but_keeps_target():
    target_mass = 444.103807533379
    tolerance = target_mass * 10e-6
    dead_end = (
        "CC21=OO1.OOcc[C@@H]Cc[C@@H]O[C@@H][C@@H][C@H]C1CCOOO[C@H]OcO[C@@H]C[C@@H]c12"
    )
    target = "COc1cc(C2C3(O)C(O)C4CC2(O)C(O)(C(=O)O4)C3C(=O)c2ccccc2)oc(=O)c1"

    assert not _has_reachable_exact_mass(_scan(dead_end), target_mass, 4.0, tolerance)
    assert _has_reachable_exact_mass(_scan(target), target_mass, 4.0, tolerance)
