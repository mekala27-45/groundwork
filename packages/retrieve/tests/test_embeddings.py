"""Embedding backend tests: the tfidf fallback's own properties (fixed
dimension, l2 normalization, batch independence) and the auto probe's
resolution logic. LocalModelEmbedder needs a real huggingface.co fetch to
load model weights the first time, so its one test is skipped unless this
environment can actually reach the hub, the same shape as
test_ocr_fallback.py skipping tesseract dependent tests rather than
mocking around a capability the codebase treats as optional.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import pytest

from groundwork_core.network import can_reach_huggingface
from groundwork_retrieve.embeddings import (
    EMBEDDING_DIM,
    LocalModelEmbedder,
    TfidfEmbedder,
    get_embedder,
    reset_embedder_cache,
    resolve_embedding_backend,
)

requires_huggingface = pytest.mark.skipif(
    not can_reach_huggingface(), reason="huggingface.co is not reachable from this environment"
)


@pytest.fixture(autouse=True)
def _reset_embedder_cache() -> Iterator[None]:
    reset_embedder_cache()
    yield
    reset_embedder_cache()


def _norm(vector: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vector))


def test_tfidf_embedder_returns_fixed_dimension_l2_normalized_vectors() -> None:
    embedder = TfidfEmbedder()

    vectors = embedder.embed(["the quick brown fox", "a completely different sentence"])

    assert len(vectors) == 2
    for vector in vectors:
        assert len(vector) == EMBEDDING_DIM
        assert math.isclose(_norm(vector), 1.0, abs_tol=1e-4)


def test_tfidf_embedder_empty_input_returns_empty_output() -> None:
    assert TfidfEmbedder().embed([]) == []


def test_tfidf_embedder_degenerate_text_is_a_safe_zero_vector() -> None:
    # HashingVectorizer's default word pattern requires two or more word
    # characters per token, so an empty string has nothing to hash. That
    # must come back as a valid zero vector, never a crash or a nan.
    [vector] = TfidfEmbedder().embed([""])
    assert len(vector) == EMBEDDING_DIM
    assert all(v == 0.0 for v in vector)


def test_tfidf_embedder_is_independent_of_what_else_is_batched_with_it() -> None:
    """The whole reason the fallback drops idf: embed() must be a pure
    function of its own text, not of whichever other texts happen to share
    a batch with it (see embeddings.py's module docstring)."""
    embedder = TfidfEmbedder()
    text = "the onboarding process begins with a discovery call"

    [alone] = embedder.embed([text])
    [with_company, _other] = embedder.embed(
        [text, "an entirely unrelated sentence about filing taxes"]
    )

    assert alone == with_company


def test_tfidf_embedder_same_text_repeated_in_one_batch_matches() -> None:
    [first, second] = TfidfEmbedder().embed(["repeat me exactly", "repeat me exactly"])
    assert first == second


@pytest.mark.parametrize("choice", ["local_model", "tfidf"])
def test_resolve_embedding_backend_honors_explicit_override(
    monkeypatch: pytest.MonkeyPatch, choice: str
) -> None:
    monkeypatch.setenv("GROUNDWORK_EMBEDDING_BACKEND", choice)
    assert resolve_embedding_backend() == choice


def test_resolve_embedding_backend_auto_uses_local_model_when_huggingface_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUNDWORK_EMBEDDING_BACKEND", "auto")
    monkeypatch.setattr("groundwork_retrieve.embeddings.can_reach_huggingface", lambda: True)

    assert resolve_embedding_backend() == "local_model"


def test_resolve_embedding_backend_auto_falls_back_to_tfidf_when_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUNDWORK_EMBEDDING_BACKEND", "auto")
    monkeypatch.setattr("groundwork_retrieve.embeddings.can_reach_huggingface", lambda: False)

    assert resolve_embedding_backend() == "tfidf"


def test_get_embedder_caches_the_constructed_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUNDWORK_EMBEDDING_BACKEND", "tfidf")

    assert get_embedder() is get_embedder()


def test_get_embedder_returns_a_tfidf_instance_for_the_tfidf_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUNDWORK_EMBEDDING_BACKEND", "tfidf")

    assert isinstance(get_embedder(), TfidfEmbedder)


@requires_huggingface
def test_local_model_embedder_produces_the_right_dimension_from_a_real_model() -> None:
    embedder = LocalModelEmbedder()
    [vector] = embedder.embed(["a real sentence embedded by the real model"])
    assert len(vector) == EMBEDDING_DIM
