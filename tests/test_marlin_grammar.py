import torch

from marlin.grammar import (
    SafeGrammarMask,
    _has_reachable_exact_mass,
    _has_structurally_viable_continuation,
    _has_vocabulary_completion,
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


def test_safe_grammar_rejects_bond_order_valence_overflow():
    assert _scan("C#C#C") is None
    assert _scan("C=C=C") is not None
    assert _scan("[NH3]#C") is None


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
    assert _scan("[2H]").terminal
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


def test_safe_grammar_accepts_hypervalent_phosphorus_and_sulfur():
    assert _scan("OP(=O)(O)O").terminal
    assert _scan("CS(=O)(=O)C").terminal


def test_safe_grammar_retains_all_valid_tokens_and_masks_invalid_token():
    tokens = ("[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "C", "O", "1")
    grammar = SafeGrammarMask(
        tokens,
        lambda ids: "".join(tokens[index] for index in ids if index >= 5),
        eos_token_id=2,
        mask_token_id=4,
        special_token_ids=(0, 1, 2, 3, 4),
    )
    logits = torch.full((len(tokens),), -10.0)
    logits[7] = 10.0
    logits[5] = 9.0
    logits[6] = 8.0

    constrained = grammar([], logits)

    assert torch.isneginf(constrained[7])
    assert torch.isfinite(constrained[5:7]).all()
    assert int(constrained.argmax()) == 5


def test_safe_grammar_does_not_false_prune_across_partial_block_hole():
    tokens = ("[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "C", "O")
    grammar = SafeGrammarMask(
        tokens,
        lambda ids: "".join(tokens[index] for index in ids if index >= 5),
        eos_token_id=2,
        mask_token_id=4,
        special_token_ids=(0, 1, 2, 3, 4),
    )
    logits = torch.arange(len(tokens), dtype=torch.float32)

    constrained = grammar([1, 4], logits)

    assert torch.isneginf(constrained[2])
    assert torch.equal(constrained[:2], logits[:2])
    assert torch.equal(constrained[3:], logits[3:])


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


def test_safe_grammar_does_not_apply_an_unproven_mass_reachability_prune():
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

    assert constrained[5] == 1.0


def test_safe_grammar_prunes_invalid_dead_end_but_keeps_target():
    target_mass = 444.103807533379
    tolerance = target_mass * 10e-6
    dead_end = (
        "CC21=OO1.OOcc[C@@H]Cc[C@@H]O[C@@H][C@@H][C@H]C1CCOOO[C@H]OcO[C@@H]C[C@@H]c12"
    )
    target = "COc1cc(C2C3(O)C(O)C4CC2(O)C(O)(C(=O)O4)C3C(=O)c2ccccc2)oc(=O)c1"

    assert _scan(dead_end) is None
    assert _has_reachable_exact_mass(_scan(target), target_mass, 4.0, tolerance)


def test_safe_grammar_prunes_unclosed_stranded_structure():
    target_mass = 444.103807533379
    tolerance = target_mass * 10e-6
    stranded = (
        "O[C@H][C@@H]Occ([C@@H]Cc[C@][C@@H]7O[C@@H][C@@H]cccC[C@@H]"
        "1%13OCOC[C@@H]C%116(O1))7"
    )
    target = "COc1cc(C2C3(O)C(O)C4CC2(O)C(O)(C(=O)O4)C3C(=O)c2ccccc2)oc(=O)c1"
    stranded_state = _scan(stranded)
    target_state = _scan(target)

    assert stranded_state is None
    assert _has_structurally_viable_continuation(
        target, target_state, target_mass, 4.0, tolerance
    )


def test_safe_grammar_keeps_incomplete_bracket_and_bonded_ring_closure():
    target_mass = 84.093900384
    tolerance = target_mass * 10e-6

    assert _has_structurally_viable_continuation(
        "[N", _scan("[N"), target_mass, 4.0, tolerance
    )
    assert _has_structurally_viable_continuation(
        "C1CCCCC-", _scan("C1CCCCC-"), target_mass, 4.0, tolerance
    )


def test_safe_grammar_treats_open_ring_atoms_as_active_hydrogen_sites():
    open_ring = _scan("C1CCCCC")
    closed_ring = _scan("C1CCCCC1")

    assert open_ring.hydrogen_bounds(0)[0] < closed_ring.hydrogen_bounds(0)[0]


def test_safe_grammar_requires_partial_atom_completion_in_vocabulary():
    target_mass = 444.103807533379
    tolerance = target_mass * 10e-6
    partial = "C" * 10 + "[13C-12"

    assert not _has_vocabulary_completion(
        partial, ("C", "O", "1", "[C]"), target_mass, 4.0, tolerance
    )
    assert _has_vocabulary_completion(partial, ("]",), target_mass, 4.0, tolerance)
