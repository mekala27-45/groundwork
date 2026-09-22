"""split_claims tests. Pure text parsing, no database involved, so every
test here runs unconditionally.
"""

from __future__ import annotations

from groundwork_core.ids import new_id
from groundwork_verify.split import split_claims

CHUNK_IDS = [new_id(), new_id(), new_id()]


def test_single_sentence_with_a_marker_resolves_its_citation() -> None:
    claims = split_claims("Sessions run fifty minutes. [1]", CHUNK_IDS)

    assert len(claims) == 1
    assert claims[0].text == "Sessions run fifty minutes."
    assert claims[0].cited_chunk_id == CHUNK_IDS[0]


def test_multiple_sentences_each_with_their_own_marker() -> None:
    answer = "Sessions run fifty minutes. [1] Foundations costs four hundred fifty dollars. [2]"

    claims = split_claims(answer, CHUNK_IDS)

    assert len(claims) == 2
    assert claims[0].cited_chunk_id == CHUNK_IDS[0]
    assert claims[1].cited_chunk_id == CHUNK_IDS[1]


def test_a_marker_backward_fills_unmarked_sentences_before_it() -> None:
    """A citation commonly closes out a short run of sentences making one
    point together, not only the clause it is typeset next to."""
    answer = "Sessions run fifty minutes. They are held over video call. [1]"

    claims = split_claims(answer, CHUNK_IDS)

    assert len(claims) == 2
    assert claims[0].text == "Sessions run fifty minutes."
    assert claims[0].cited_chunk_id == CHUNK_IDS[0]
    assert claims[1].text == "They are held over video call."
    assert claims[1].cited_chunk_id == CHUNK_IDS[0]


def test_backward_fill_stops_at_the_previous_marker() -> None:
    answer = (
        "Sessions run fifty minutes. [1] They are unrelated to billing. Billing is monthly. [2]"
    )

    claims = split_claims(answer, CHUNK_IDS)

    assert [c.cited_chunk_id for c in claims] == [
        CHUNK_IDS[0],
        CHUNK_IDS[1],
        CHUNK_IDS[1],
    ]


def test_a_trailing_run_with_no_marker_anywhere_after_it_has_no_citation() -> None:
    answer = "Sessions run fifty minutes. [1] This closing sentence cites nothing."

    claims = split_claims(answer, CHUNK_IDS)

    assert claims[-1].text == "This closing sentence cites nothing."
    assert claims[-1].cited_chunk_id is None


def test_an_answer_with_no_marker_at_all_has_no_citations() -> None:
    claims = split_claims("This document does not cover that question.", CHUNK_IDS)

    assert len(claims) == 1
    assert claims[0].cited_chunk_id is None


def test_a_marker_index_outside_cited_chunk_ids_range_resolves_to_no_citation() -> None:
    """A hallucinated or off by one citation number is exactly the kind
    of thing this layer exists to surface, not crash on."""
    claims = split_claims("A claim citing a number that does not exist. [99]", CHUNK_IDS)

    assert claims[0].cited_chunk_id is None


def test_multiple_markers_in_one_sentence_uses_the_last_one() -> None:
    claims = split_claims("Supported by two passages at once. [1][2]", CHUNK_IDS)

    assert claims[0].cited_chunk_id == CHUNK_IDS[1]


def test_citation_markers_are_stripped_from_the_claim_text() -> None:
    claims = split_claims("Sessions run fifty minutes. [1]", CHUNK_IDS)

    assert "[1]" not in claims[0].text


def test_an_answer_that_is_only_a_marker_with_no_surrounding_text_returns_no_claims() -> None:
    """A degenerate fragment that is nothing but a citation marker has no
    claim text left once the marker is stripped, so it contributes
    nothing to score, rather than an empty-string "claim" no scorer could
    meaningfully judge."""
    assert split_claims("[1]", CHUNK_IDS) == []


def test_empty_answer_returns_no_claims() -> None:
    assert split_claims("", CHUNK_IDS) == []


def test_whitespace_only_answer_returns_no_claims() -> None:
    assert split_claims("   \n\n  ", CHUNK_IDS) == []
