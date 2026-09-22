"""is_out_of_scope_refusal tests. Pure function, no database involved, so
every test here runs unconditionally.
"""

from __future__ import annotations

from groundwork_generate.generate import NOT_COVERED_MESSAGE
from groundwork_verify.refusal import is_out_of_scope_refusal


def test_the_exact_not_covered_message_with_no_citations_passes() -> None:
    assert is_out_of_scope_refusal(NOT_COVERED_MESSAGE, cited_chunk_count=0) is True


def test_a_citation_alongside_the_refusal_message_fails() -> None:
    """A real refusal never has a citation to begin with (every path that
    produces NOT_COVERED_MESSAGE also produces an empty cited_chunk_ids),
    but this function checks both conditions independently rather than
    trusting that invariant blindly."""
    assert is_out_of_scope_refusal(NOT_COVERED_MESSAGE, cited_chunk_count=1) is False


def test_a_confident_wrong_answer_that_merely_contains_the_word_not_fails() -> None:
    """The deliberate violation case the module docstring warns about: a
    softer, word-matching heuristic would wrongly pass this."""
    answer = "This is not a small detail: Meridian coaching costs $450 a month."
    assert is_out_of_scope_refusal(answer, cited_chunk_count=0) is False


def test_a_similar_but_not_exact_refusal_phrasing_fails() -> None:
    """Deterministic means exact, not merely refusal shaped: a real LLM
    asked to refuse in its own words would not match this, a documented,
    unmeasured gap (see the module docstring) rather than a silent one."""
    answer = "I'm sorry, but this document does not cover that question."
    assert is_out_of_scope_refusal(answer, cited_chunk_count=0) is False


def test_surrounding_whitespace_around_an_otherwise_exact_message_still_passes() -> None:
    assert is_out_of_scope_refusal(f"  {NOT_COVERED_MESSAGE}\n", cited_chunk_count=0) is True


def test_an_empty_answer_fails() -> None:
    assert is_out_of_scope_refusal("", cited_chunk_count=0) is False
