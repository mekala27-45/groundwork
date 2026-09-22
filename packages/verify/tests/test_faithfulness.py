"""Faithfulness scoring tests: LexicalFaithfulnessScorer's own properties
(the same shape test_rerank.py uses for LexicalReranker), the auto probe's
resolution logic, and check_faithfulness' orchestration, including the
extractive short circuit. No database is involved anywhere in this file.
CrossEncoderFaithfulnessScorer needs a real huggingface.co fetch to load
model weights, so its one test is skipped unless this environment can
actually reach the hub.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from groundwork_api.models import Chunk, ChunkStrategy, NliLabel
from groundwork_core.ids import new_id
from groundwork_core.network import can_reach_huggingface
from groundwork_verify.faithfulness import (
    CrossEncoderFaithfulnessScorer,
    LexicalFaithfulnessScorer,
    check_faithfulness,
    get_faithfulness_scorer,
    reset_faithfulness_scorer_cache,
    resolve_faithfulness_backend,
)

requires_huggingface = pytest.mark.skipif(
    not can_reach_huggingface(), reason="huggingface.co is not reachable from this environment"
)


@pytest.fixture(autouse=True)
def _reset_faithfulness_scorer_cache() -> Iterator[None]:
    reset_faithfulness_scorer_cache()
    yield
    reset_faithfulness_scorer_cache()


def _chunk(text: str) -> Chunk:
    return Chunk(
        document_id=new_id(),
        workspace_id=new_id(),
        strategy=ChunkStrategy.NAIVE,
        text=text,
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(text),
    )


# -- LexicalFaithfulnessScorer --------------------------------------------


def test_lexical_scorer_labels_a_high_overlap_claim_entailed() -> None:
    label, score = LexicalFaithfulnessScorer().score(
        "Sessions run fifty minutes.", "Coaching sessions run fifty minutes over video call."
    )

    assert label == NliLabel.ENTAILED
    assert score > 0.0


def test_lexical_scorer_labels_a_negation_flip_on_high_overlap_contradicted() -> None:
    label, _score = LexicalFaithfulnessScorer().score(
        "Coaching is a substitute for therapy.",
        "Coaching is not a substitute for therapy or mental health treatment.",
    )

    assert label == NliLabel.CONTRADICTED


def test_lexical_scorer_labels_a_low_overlap_claim_unsupported() -> None:
    label, _score = LexicalFaithfulnessScorer().score(
        "The moon is made of cheese.", "Sessions run fifty minutes over video call."
    )

    assert label == NliLabel.UNSUPPORTED


def test_lexical_scorer_empty_claim_is_unsupported_at_zero_confidence() -> None:
    label, score = LexicalFaithfulnessScorer().score("", "some chunk text")

    assert label == NliLabel.UNSUPPORTED
    assert score == 0.0


def test_lexical_scorer_is_deterministic_across_repeated_calls() -> None:
    scorer = LexicalFaithfulnessScorer()
    args = ("Sessions run fifty minutes.", "Coaching sessions run fifty minutes.")

    assert scorer.score(*args) == scorer.score(*args)


# -- resolve_faithfulness_backend and the cached scorer -------------------


@pytest.mark.parametrize("choice", ["local_model", "lexical"])
def test_resolve_faithfulness_backend_honors_explicit_override(
    monkeypatch: pytest.MonkeyPatch, choice: str
) -> None:
    monkeypatch.setenv("GROUNDWORK_FAITHFULNESS_BACKEND", choice)
    assert resolve_faithfulness_backend() == choice


def test_resolve_faithfulness_backend_auto_uses_local_model_when_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUNDWORK_FAITHFULNESS_BACKEND", "auto")
    monkeypatch.setattr("groundwork_verify.faithfulness.can_reach_huggingface", lambda: True)

    assert resolve_faithfulness_backend() == "local_model"


def test_resolve_faithfulness_backend_auto_falls_back_to_lexical_when_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUNDWORK_FAITHFULNESS_BACKEND", "auto")
    monkeypatch.setattr("groundwork_verify.faithfulness.can_reach_huggingface", lambda: False)

    assert resolve_faithfulness_backend() == "lexical"


def test_get_faithfulness_scorer_caches_the_constructed_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUNDWORK_FAITHFULNESS_BACKEND", "lexical")
    assert get_faithfulness_scorer() is get_faithfulness_scorer()


def test_get_faithfulness_scorer_returns_a_lexical_instance_for_the_lexical_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUNDWORK_FAITHFULNESS_BACKEND", "lexical")
    assert isinstance(get_faithfulness_scorer(), LexicalFaithfulnessScorer)


# -- check_faithfulness: the extractive short circuit ----------------------


def test_check_faithfulness_extractive_answer_is_entailed_without_calling_a_scorer() -> None:
    class _ExplodingScorer:
        def score(self, claim: str, chunk_text: str) -> tuple[NliLabel, float]:
            raise AssertionError("the scorer must never be called for an extractive answer")

    top = _chunk("Sessions run fifty minutes.")
    verifications = check_faithfulness(
        "anything, extractive answers never text parse their rendered answer",
        extractive_fallback=True,
        chunks=[top],
        scorer=_ExplodingScorer(),
    )

    assert len(verifications) == 1
    assert verifications[0].label == NliLabel.ENTAILED
    assert verifications[0].score == 1.0
    assert verifications[0].claim.text == "Sessions run fifty minutes."
    assert verifications[0].claim.cited_chunk_id == top.id


def test_check_faithfulness_extractive_with_no_chunks_returns_no_claims() -> None:
    assert check_faithfulness("q", extractive_fallback=True, chunks=[]) == []


# -- check_faithfulness: the real generation path --------------------------


class _RecordingScorer:
    """Returns a fixed label for every call while recording the exact
    (claim, chunk_text) pairs it was asked to score, so a test can assert
    both that scoring actually ran (the pairs it recorded) and that the
    right chunk text reached the right claim."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def score(self, claim: str, chunk_text: str) -> tuple[NliLabel, float]:
        self.calls.append((claim, chunk_text))
        return NliLabel.ENTAILED, 0.9


async def test_check_faithfulness_routes_each_claim_to_its_cited_chunk_text() -> None:
    first = _chunk("Sessions run fifty minutes.")
    second = _chunk("Foundations costs four hundred fifty dollars per month.")
    scorer = _RecordingScorer()

    check_faithfulness(
        "Sessions run fifty minutes. [1] Foundations costs four hundred fifty dollars. [2]",
        extractive_fallback=False,
        chunks=[first, second],
        scorer=scorer,
    )

    assert scorer.calls == [
        ("Sessions run fifty minutes.", first.text),
        ("Foundations costs four hundred fifty dollars.", second.text),
    ]


def test_check_faithfulness_an_uncited_claim_is_unsupported_without_calling_the_scorer() -> None:
    scorer = _RecordingScorer()

    verifications = check_faithfulness(
        "This closing sentence cites nothing at all.",
        extractive_fallback=False,
        chunks=[_chunk("some chunk text")],
        scorer=scorer,
    )

    assert verifications[0].label == NliLabel.UNSUPPORTED
    assert verifications[0].score == 0.0
    assert scorer.calls == []


def test_check_faithfulness_uses_the_process_default_scorer_when_none_is_passed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUNDWORK_FAITHFULNESS_BACKEND", "lexical")
    chunk = _chunk("Sessions run fifty minutes over video call.")

    verifications = check_faithfulness(
        "Sessions run fifty minutes. [1]", extractive_fallback=False, chunks=[chunk]
    )

    assert verifications[0].label == NliLabel.ENTAILED


@requires_huggingface
def test_cross_encoder_faithfulness_scorer_entails_a_supported_claim_with_a_real_model() -> None:
    scorer = CrossEncoderFaithfulnessScorer()

    label, _score = scorer.score(
        "Paris is the capital of France.",
        "Paris is the capital and largest city of France, situated on the river Seine.",
    )

    assert label == NliLabel.ENTAILED
