"""Lock the classifications the dead-end diagnosis reports."""

from rdkit import Chem

from marlin.grammar import _scan
from scripts.diagnose_marlin_dead_ends import (
    classify_parse_failure,
    group_of,
    phantom_atoms,
)


def _row(**overrides):
    row = {
        "attempts": 8,
        "candidate_returned": False,
        "constraint_dead_ends": 0,
        "valid": 0,
    }
    row.update(overrides)
    return row


def test_group_of_separates_the_four_outcomes():
    assert group_of(_row(candidate_returned=True, valid=3)) == "returned"
    assert group_of(_row(constraint_dead_ends=8)) == "all_dead_end"
    assert group_of(_row(constraint_dead_ends=4, valid=2)) == "mass_miss"
    assert group_of(_row(constraint_dead_ends=4, valid=0)) == "no_parse"


def test_phantom_atoms_finds_nothing_left_for_the_grammar_to_weigh_at_zero():
    # The diagnosis measured a bracket hydrogen and every element outside an
    # 18-entry table entering the state at mass 0.0 while the token table charged
    # their real mass, in 2,240 of 3,093 dead-end prefixes. Both models now ask
    # marlin.token_properties.atom_mass, so this prefix holds no weightless atom
    # where it used to hold two.
    # The ring labels carry the same three atoms as the original "C.[H].[Og]",
    # which the mask now refuses for leaving its fragments unattachable.
    counts = phantom_atoms(_scan("C12.[H]1.[Og]2"))

    assert counts["atoms"] == 3
    assert counts["zero_mass_atoms"] == 0
    assert counts["bracket_hydrogen_atoms"] == 0
    assert counts["foreign_element_atoms"] == 0
    assert phantom_atoms(_scan("CCO"))["zero_mass_atoms"] == 0


def test_classify_parse_failure_names_the_rdkit_cause():
    assert classify_parse_failure("CCO")["bucket"] == "parses"
    assert classify_parse_failure("c1cccc1")["bucket"] == "kekulization"
    # The diagnosis recorded 33 of 91 unparsable terminal strings as a ring closure
    # duplicating an existing bond, which the grammar accepted and RDKit refused. The
    # grammar now refuses it too, so the classifier is exercised on the string alone.
    assert _scan("C12CC12") is None
    assert Chem.MolFromSmiles("C12CC12") is None
    assert classify_parse_failure("C12CC12")["bucket"] == "duplicate_ring_bond"
