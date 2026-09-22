"""Reranker tests: BM25's own properties (relevance ordering, term
frequency, determinism, edge cases) and the auto probe's resolution logic,
the same shape as test_embeddings.py. CrossEncoderReranker needs a real
huggingface.co fetch to load model weights, so its one test is skipped
unless this environment can actually reach the hub.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from groundwork_core.network import can_reach_huggingface
from groundwork_retrieve.rerank import (
    CrossEncoderReranker,
    LexicalReranker,
    get_reranker,
    reset_reranker_cache,
    resolve_rerank_backend,
)

requires_huggingface = pytest.mark.skipif(
    not can_reach_huggingface(), reason="huggingface.co is not reachable from this environment"
)


@pytest.fixture(autouse=True)
def _reset_reranker_cache() -> Iterator[None]:
    reset_reranker_cache()
    yield
    reset_reranker_cache()


def test_lexical_reranker_scores_the_more_relevant_candidate_higher() -> None:
    reranker = LexicalReranker()

    scores = reranker.rerank(
        "python programming tutorial",
        [
            "a detailed python programming tutorial for beginners",
            "a recipe for chocolate cake",
        ],
    )

    assert scores[0] > scores[1]


def test_lexical_reranker_returns_one_score_per_candidate_in_the_same_order() -> None:
    reranker = LexicalReranker()
    candidates = ["alpha document", "beta document", "gamma document"]

    scores = reranker.rerank("alpha", candidates)

    assert len(scores) == len(candidates)
    assert scores[0] > scores[1] == scores[2]


def test_lexical_reranker_empty_candidates_returns_empty_scores() -> None:
    assert LexicalReranker().rerank("query", []) == []


def test_lexical_reranker_empty_query_scores_every_candidate_zero() -> None:
    scores = LexicalReranker().rerank("", ["anything", "something else"])
    assert scores == [0.0, 0.0]


def test_lexical_reranker_query_term_absent_from_every_candidate_does_not_crash() -> None:
    scores = LexicalReranker().rerank("python zzzznotarealword", ["python basics", "snake facts"])
    assert len(scores) == 2
    assert all(score >= 0.0 for score in scores)


def test_lexical_reranker_more_repetitions_of_the_query_term_scores_higher() -> None:
    """BM25's term frequency component: saturating, but still monotonic
    for these two candidate lengths, confirmed by hand and checked here
    rather than only asserted from the formula."""
    scores = LexicalReranker().rerank("python", ["python", "python python python"])
    assert scores[1] > scores[0]


def test_lexical_reranker_is_deterministic_across_repeated_calls() -> None:
    reranker = LexicalReranker()
    candidates = ["python tutorial for beginners", "cake recipe"]

    first = reranker.rerank("python tutorial", candidates)
    second = reranker.rerank("python tutorial", candidates)

    assert first == second


def test_lexical_reranker_statistics_are_scoped_to_one_call_not_shared_across_calls() -> None:
    """Unlike embeddings.py's fallback, this one is allowed to use
    document frequency computed over its own batch (see module docstring)
    precisely because every call is a fresh, self contained batch: calling
    it twice with different candidate sets must not let the first call's
    statistics leak into the second."""
    reranker = LexicalReranker()

    first_call = reranker.rerank("python", ["python basics", "unrelated text here"])
    second_call = reranker.rerank("python", ["python basics", "python python python python"])

    # The score for the identical first candidate must depend only on the
    # candidates present in that call, not on a different candidate set
    # scored moments earlier.
    assert first_call[0] != second_call[0]


@pytest.mark.parametrize("choice", ["local_model", "lexical", "none"])
def test_resolve_rerank_backend_honors_explicit_override(
    monkeypatch: pytest.MonkeyPatch, choice: str
) -> None:
    monkeypatch.setenv("GROUNDWORK_RERANK_BACKEND", choice)
    assert resolve_rerank_backend() == choice


def test_resolve_rerank_backend_auto_uses_local_model_when_huggingface_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUNDWORK_RERANK_BACKEND", "auto")
    monkeypatch.setattr("groundwork_retrieve.rerank.can_reach_huggingface", lambda: True)

    assert resolve_rerank_backend() == "local_model"


def test_resolve_rerank_backend_auto_falls_back_to_lexical_when_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUNDWORK_RERANK_BACKEND", "auto")
    monkeypatch.setattr("groundwork_retrieve.rerank.can_reach_huggingface", lambda: False)

    assert resolve_rerank_backend() == "lexical"


def test_get_reranker_caches_the_constructed_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUNDWORK_RERANK_BACKEND", "lexical")
    assert get_reranker() is get_reranker()


def test_get_reranker_returns_a_lexical_instance_for_the_lexical_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUNDWORK_RERANK_BACKEND", "lexical")
    assert isinstance(get_reranker(), LexicalReranker)


def test_get_reranker_returns_none_for_the_none_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """None means "skip reranking", not "some default reranker was
    substituted in"; a caller checking for None is the whole point."""
    monkeypatch.setenv("GROUNDWORK_RERANK_BACKEND", "none")
    assert get_reranker() is None


@requires_huggingface
def test_cross_encoder_reranker_scores_the_relevant_pair_higher_with_a_real_model() -> None:
    reranker = CrossEncoderReranker()

    scores = reranker.rerank(
        "what is the capital of France?",
        [
            "Paris is the capital and largest city of France.",
            "Bananas are a good source of potassium.",
        ],
    )

    assert scores[0] > scores[1]
