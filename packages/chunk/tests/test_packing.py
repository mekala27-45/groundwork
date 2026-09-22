"""Direct tests of pack_atom_range and _join_atoms, the algorithm both
chunking strategies share, at the Atom level rather than through a full
ExtractedDocument. The property tests in test_chunking_properties.py cover
the general guarantees across many generated inputs; this file pins down
specific edge cases by hand, including a named regression test for the
separator bug hypothesis found while writing them.
"""

from __future__ import annotations

import pytest

from groundwork_chunk.models import Atom
from groundwork_chunk.packing import pack_atom_range
from groundwork_chunk.tokenize import count_tokens


@pytest.fixture(autouse=True)
def _force_approximate_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    # These tests hand trace exact token counts against the approximate
    # backend's ceil(len/4) rule, so they need to run the same way
    # regardless of whether this environment can reach tiktoken.
    monkeypatch.setenv("GROUNDWORK_TOKENIZER_BACKEND", "approximate")


def _word(text: str, *, page: int = 1) -> Atom:
    return Atom(kind="word", text=text, page_number=page)


def _table_atom(text: str, *, page: int = 1) -> Atom:
    return Atom(kind="table", text=text, page_number=page)


def test_empty_atoms_returns_no_ranges() -> None:
    assert pack_atom_range([], chunk_size_tokens=10, overlap_tokens=0) == []


def test_a_single_word_over_budget_is_still_returned_whole() -> None:
    # "elephant" is 8 characters, 2 tokens under the approximate backend,
    # already over a budget of 1 on its own with nothing else in the atom
    # sequence to blame. pack_atom_range must still make progress rather
    # than loop forever or refuse to emit anything.
    atoms = [_word("elephant")]
    assert count_tokens("elephant") > 1

    ranges = pack_atom_range(atoms, chunk_size_tokens=1, overlap_tokens=0)

    assert ranges == [(0, 1)]


def test_a_word_and_table_that_fit_separately_are_not_bundled_past_budget() -> None:
    """Regression test for the separator bug: pack_atom_range's own budget
    check once joined candidate atoms with a bare space, while the text a
    chunk actually resolves to (via _join_atoms, also used by
    resolve_chunk_text) puts a blank line around a table. "a" and "00" each
    fit a budget of 1 token on their own, but "a" + blank line + "00" does
    not, so they must land in separate chunks rather than one that quietly
    exceeds the budget once resolved.
    """
    atoms = [_word("a"), _table_atom("00")]
    assert count_tokens("a") == 1
    assert count_tokens("00") == 1
    assert count_tokens("a\n\n00") == 2  # the blank-line joined text is over budget

    ranges = pack_atom_range(atoms, chunk_size_tokens=1, overlap_tokens=0)

    assert ranges == [(0, 1), (1, 2)]


def test_a_table_alone_over_budget_is_still_returned_whole() -> None:
    atoms = [_word("intro"), _table_atom("a long table row that alone exceeds the budget")]

    ranges = pack_atom_range(atoms, chunk_size_tokens=1, overlap_tokens=0)

    # the table never gets split, even though it alone is well over budget
    assert (1, 2) in ranges


def test_overlap_never_seeds_the_next_chunk_with_a_trailing_table() -> None:
    # "aaaa" plus the table fits a budget of 2; adding "zzzz" on top does
    # not, so the first chunk closes with the table as its last atom. A
    # generous overlap_tokens=10 would, if tables were eligible, walk the
    # table straight back into the next chunk's seed; atoms[k].kind ==
    # "word" in the overlap walk-back is what is supposed to stop that.
    atoms = [_word("aaaa"), _table_atom("t1"), _word("zzzz")]

    ranges = pack_atom_range(atoms, chunk_size_tokens=2, overlap_tokens=10)

    assert ranges[0] == (0, 2)
    assert ranges[1][0] == 2, "the chunk after one ending in a table must not repeat the table"
