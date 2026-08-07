from functools import lru_cache
from pathlib import Path

import pytest
import torch

from marlin.grammar import (
    SafeGrammarMask,
    _GrammarState,
    _has_reachable_exact_mass,
    _has_structurally_viable_continuation,
    _has_vocabulary_completion,
    _has_mass_viable_atom_completion,
    _has_mass_viable_bracket_completion,
    _has_mass_viable_percent_completion,
    _scan,
    _scan_base,
    _scan_continuation,
)
from marlin.token_properties import foreign_element_token_ids, isotope_token_ids
from marlin.tokenizer import load_safe_tokenizer


TOKENIZER_PATHS = (
    Path(
        "/mnt/netstorage/nikolenko/marlin/cache/runtime-inputs-spectrum-v1/"
        "16b1af5276034c041e85a4b7c43129a790b4fc091826485b691c93f9f7b699b3/"
        "tokenizer.json"
    ),
    Path(
        "/home/nikolenko/work/Projects/MARLIN_reproduction_20260717/"
        "data/safe-gpt/tokenizer.json"
    ),
)
# Prefixes the decoder actually reaches, plus every shape whose parse depends on
# characters a candidate token supplies: a bare "B"/"C" a token can turn into
# "Br"/"Cl", an unterminated bracket, and a "%" waiting for its two digits.
EQUIVALENCE_PREFIXES = (
    "",
    "C",
    "CB",
    "CCl",
    "COc1cc(",
    "COc1cc(C",
    "COc1cc(C2C3(O)C(O)C4CC2(O)",
    "COc1cc(C2C3(O)C(O)C4CC2(O)C(O)(C(=O)O4)C3C(=O)c2ccccc2)oc(=O)c1",
    "CCO[",
    "C[13C",
    "C%1",
    "C%12",
    "C1CCCCC-",
    "CN(C)C(=O)",
    "C1CCCCC1.C",
    "[NH",
)


@lru_cache(maxsize=1)
def real_vocabulary() -> tuple[str, ...]:
    for path in TOKENIZER_PATHS:
        if path.exists():
            tokenizer = load_safe_tokenizer(path)
            return tuple(
                tokenizer.convert_ids_to_tokens(index)
                for index in range(len(tokenizer))
            )
    pytest.skip(f"no real SAFE tokenizer under {TOKENIZER_PATHS}")


def scan_signature(state: _GrammarState | None):
    if state is None:
        return None
    return (
        state.terminal,
        state.incomplete_token,
        sum(state.atom_masses.values()),
        state.hydrogen_bounds(4.0),
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


def test_safe_grammar_rejects_position_after_partial_block_hole():
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

    assert torch.isneginf(constrained).all()


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


def _reachability_grammar(prefix: str, tokens: tuple[str, ...]) -> SafeGrammarMask:
    return SafeGrammarMask(
        tokens,
        lambda _ids: prefix,
        eos_token_id=2,
        special_token_ids=(0, 1, 2, 3, 4),
        mass_reachability_prune=True,
    )


def test_mass_reachability_prune_removes_tokens_that_cannot_reach_target_mass():
    tokens = ("[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "1", "C")
    grammar = _reachability_grammar("C" * 32, tokens)
    logits = torch.zeros(len(tokens))

    constrained = grammar([], logits, 444.103807533379)

    assert torch.isneginf(constrained[5])
    assert torch.isneginf(constrained[6])


def test_mass_reachability_prune_keeps_tokens_on_a_reachable_path():
    tokens = ("[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "1", "C")
    grammar = _reachability_grammar("CC", tokens)
    logits = torch.zeros(len(tokens))

    constrained = grammar([], logits, 444.103807533379)

    assert constrained[6] == 0.0


def test_mass_reachability_prune_forbids_eos_when_mass_misses_the_target():
    tokens = ("[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "1", "C")
    grammar = _reachability_grammar("CC", tokens)
    logits = torch.zeros(len(tokens))

    constrained = grammar([], logits, 444.103807533379)

    assert torch.isneginf(constrained[2])


def test_mass_reachability_prune_allows_eos_when_hydrogens_close_the_target():
    target = "COc1cc(C2C3(O)C(O)C4CC2(O)C(O)(C(=O)O4)C3C(=O)c2ccccc2)oc(=O)c1"
    tokens = ("[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "1", "C")
    grammar = _reachability_grammar(target, tokens)
    logits = torch.zeros(len(tokens))

    constrained = grammar([], logits, 444.103807533379)

    assert constrained[2] == 0.0


def _syntax_grammar(prefix: str, tokens: tuple[str, ...]) -> SafeGrammarMask:
    return SafeGrammarMask(
        tokens,
        lambda _ids: prefix,
        eos_token_id=2,
        special_token_ids=(0, 1, 2, 3, 4),
    )


def test_safe_grammar_rejects_bracket_digits_that_no_element_can_complete():
    # A carbonyl oxygen carrying a branch admits no element, so the isotope digits
    # opened by "[" lead to a trap state the decoder can never leave.
    tokens = ("[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "6", "7", "C", "O", "Br]")
    grammar = _syntax_grammar("CC(C)=O(-[", tokens)

    constrained = grammar([], torch.zeros(len(tokens)))

    assert bool(torch.isinf(constrained).all())


def test_safe_grammar_keeps_bracket_digits_that_an_element_can_complete():
    # "1" opens the isotope of [13C]; it survives only while the vocabulary can
    # still finish it.
    completable = ("[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "1", "3C]", "C]")
    grammar = _syntax_grammar("CCO[", completable)

    constrained = grammar([], torch.zeros(len(completable)))

    assert constrained[5] == 0.0
    assert constrained[7] == 0.0


def test_safe_grammar_drops_bracket_digits_once_the_vocabulary_cannot_finish_them():
    incompletable = ("[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "1", "C]")
    grammar = _syntax_grammar("CCO[", incompletable)

    constrained = grammar([], torch.zeros(len(incompletable)))

    assert bool(torch.isinf(constrained[5]))
    assert constrained[6] == 0.0


@pytest.mark.parametrize("prefix", EQUIVALENCE_PREFIXES)
def test_incremental_scan_matches_a_full_rescan_of_every_real_token(prefix):
    # The mask scans the prefix once and advances a copy over each candidate
    # token. Nothing in the decoder would reveal a divergence from the full
    # rescan, so pin it over the whole vocabulary.
    vocabulary = real_vocabulary()
    base = _scan_base(prefix)

    assert base is not None
    for token in vocabulary:
        assert scan_signature(_scan_continuation(base, token)) == scan_signature(
            _scan(prefix + token)
        ), f"{prefix!r} + {token!r}"


def test_scan_base_defers_a_tail_that_a_token_can_reinterpret():
    # "B" alone is boron, but the vocabulary holds "r", so the base must leave
    # the trailing character to the continuation.
    boron = _scan_base("CB")
    bromine = _scan_continuation(boron, "r")

    assert boron.pending == "B"
    assert sum(boron.state.atom_masses.values()) == sum(_scan("C").atom_masses.values())
    assert sum(bromine.atom_masses.values()) == sum(_scan("CBr").atom_masses.values())
    assert _scan_base("CCO[").pending == "["
    assert _scan_base("C%1").pending == "%1"
    assert _scan_base("C%12").pending == ""


def test_grammar_state_copy_shares_no_mutable_container():
    state = _scan("C1CCCCC(N)[13CH3]")
    clone = state.copy()

    # Equality over the whole __dict__ catches a field the clone forgot to carry.
    assert vars(clone) == vars(state)
    for name, value in vars(clone).items():
        if isinstance(value, (dict, list)):
            assert value is not getattr(state, name), name
    clone.atom_masses[99] = 1.0
    clone.branch_atoms.append(99)
    clone.open_rings["9"] = 99

    assert 99 not in state.atom_masses
    assert 99 not in state.branch_atoms
    assert "9" not in state.open_rings


def test_forbidden_token_ids_withhold_support_without_touching_neighbours():
    tokens = ("[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "C", "O", "[13C]")
    decode = lambda ids: "".join(tokens[index] for index in ids if index >= 5)
    logits = torch.full((len(tokens),), 1.0)

    open_mask = SafeGrammarMask(
        tokens,
        decode,
        eos_token_id=2,
        mask_token_id=4,
        special_token_ids=(0, 1, 2, 3, 4),
    )
    closed_mask = SafeGrammarMask(
        tokens,
        decode,
        eos_token_id=2,
        mask_token_id=4,
        special_token_ids=(0, 1, 2, 3, 4),
        forbidden_token_ids=(7,),
    )

    assert torch.isfinite(open_mask([], logits)[7])
    constrained = closed_mask([], logits)
    assert torch.isneginf(constrained[7])
    # The tokens a real target actually needs keep the support they had.
    assert torch.isfinite(constrained[5:7]).all()


def test_isotope_token_ids_match_only_mass_numbered_bracket_atoms():
    tokens = ("C", "[13C]", "[nH]", "[C@@H]", "[1", "[100Tc+3]", "O", "[Na+]")

    assert isotope_token_ids(tokens) == (1, 4, 5)


def test_foreign_element_token_ids_keep_the_organic_set_and_drop_metals():
    tokens = ("C", "c", "O", "[nH]", "Cl", "Br", "[Se]", "[100Tc+3]", "[Fe+2]", "1")

    foreign = foreign_element_token_ids(tokens)

    assert [tokens[i] for i in foreign] == ["[Se]", "[100Tc+3]", "[Fe+2]"]


def test_foreign_element_token_ids_leave_unresolvable_tokens_supported():
    # A partial token whose element cannot be read must not be banned on a guess.
    assert foreign_element_token_ids(("[", "(", ")", "=", "%10")) == ()
