"""Secondary quality scoring tests. parse_quality_score is pure, no
network and no database, so those tests run unconditionally. The one real
LiteLLM call test is skip marked behind requires_llm_key, the same shape
test_generate.py, test_rerank.py and test_embeddings.py use for their own
real-model tests, since this sandbox has no LLM API key configured.
"""

from __future__ import annotations

import json

import pytest

from groundwork_core.config import get_settings
from groundwork_generate.ledger import SpendLedger
from groundwork_verify.quality import (
    Judge,
    LiteLLMJudge,
    QualityScore,
    parse_quality_score,
    resolve_quality_backend,
    score_reply_quality,
)

requires_llm_key = pytest.mark.skipif(
    not get_settings().llm_api_key, reason="no LLM API key configured in this environment"
)


class _StubJudge:
    """Records exactly what it was asked to score and returns a fixed,
    caller supplied result, the same recording-stub shape
    packages/api/tests/test_chat.py's _RecordingGenerator uses: proves
    score_reply_quality delegates rather than transforms, without any
    network access."""

    def __init__(self, result: QualityScore | None) -> None:
        self._result = result
        self.received: tuple[str, str] | None = None

    async def score(self, question: str, answer: str) -> QualityScore | None:
        self.received = (question, answer)
        return self._result


# -- parse_quality_score: pure, network free -------------------------------


def test_parse_quality_score_accepts_valid_arguments() -> None:
    raw = json.dumps({"clarity": 4, "helpfulness": 5, "rationale": "Direct and on topic."})

    score = parse_quality_score(raw)

    assert score.clarity == 4
    assert score.helpfulness == 5
    assert score.rationale == "Direct and on topic."


def test_parse_quality_score_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_quality_score("{not json")


def test_parse_quality_score_rejects_a_clarity_score_outside_one_to_five() -> None:
    raw = json.dumps({"clarity": 6, "helpfulness": 3, "rationale": "Out of range on purpose."})

    with pytest.raises(ValueError, match="did not match the schema"):
        parse_quality_score(raw)


def test_parse_quality_score_rejects_a_missing_required_field() -> None:
    raw = json.dumps({"clarity": 3, "rationale": "helpfulness is missing entirely."})

    with pytest.raises(ValueError, match="did not match the schema"):
        parse_quality_score(raw)


def test_parse_quality_score_rejects_an_extra_field_strict_model_forbids() -> None:
    """QualityScore is a StrictModel, extra='forbid': a judge that pads
    its arguments with a field nobody asked for fails validation rather
    than having the extra field silently dropped."""
    raw = json.dumps({"clarity": 3, "helpfulness": 3, "rationale": "fine", "faithfulness": 5})

    with pytest.raises(ValueError, match="did not match the schema"):
        parse_quality_score(raw)


# -- resolve_quality_backend: identical shape to resolve_generation_backend


def test_resolve_quality_backend_is_unavailable_with_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GROUNDWORK_LLM_API_KEY", raising=False)
    assert resolve_quality_backend() == "unavailable"


def test_resolve_quality_backend_is_litellm_judge_once_a_key_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GROUNDWORK_LLM_API_KEY", "sk-test-not-a-real-key")
    assert resolve_quality_backend() == "litellm_judge"


# -- score_reply_quality: the one entry point most callers should use -----


async def test_score_reply_quality_returns_exactly_what_the_judge_returns() -> None:
    expected = QualityScore(clarity=5, helpfulness=4, rationale="Clear, cites its source.")
    judge: Judge = _StubJudge(expected)

    result = await score_reply_quality("What is the fee?", "The fee is $50. [1]", judge=judge)

    assert result == expected


async def test_score_reply_quality_passes_the_question_and_answer_through_unchanged() -> None:
    stub = _StubJudge(None)

    await score_reply_quality("What is the fee?", "The fee is $50. [1]", judge=stub)

    assert stub.received == ("What is the fee?", "The fee is $50. [1]")


async def test_score_reply_quality_returns_none_when_the_judge_itself_returns_none() -> None:
    """score_reply_quality never substitutes a synthetic score for a
    judge's own None: a judge that could not produce a usable score stays
    unmeasured, not silently filled in."""
    judge: Judge = _StubJudge(None)

    result = await score_reply_quality("q", "a", judge=judge)

    assert result is None


async def test_score_reply_quality_returns_none_with_no_judge_and_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real default path in this sandbox: no explicit judge passed, no
    key configured, so resolution finds nothing to construct at all."""
    monkeypatch.delenv("GROUNDWORK_LLM_API_KEY", raising=False)

    result = await score_reply_quality("q", "a")

    assert result is None


# -- LiteLLMJudge: the part testable with no real key ----------------------


async def test_litellm_judge_returns_none_when_the_ledger_is_already_exhausted() -> None:
    """Proves the shared spend ledger is checked before any network
    access, exactly like LiteLLMGenerator's identical guard:
    constructible and callable with no real key and no network access,
    since this test never reaches the litellm import at all."""
    exhausted_ledger = SpendLedger(ceiling_usd=1.00, spent_usd=1.00)
    judge = LiteLLMJudge(model="gpt-4o-mini", ledger=exhausted_ledger)

    result = await judge.score("How long is a session?", "Sessions run fifty minutes.")

    assert result is None
    assert exhausted_ledger.spent_usd == 1.00  # unchanged: no real call was made to add to it


# -- the real network path, skipped without a configured key ---------------


@requires_llm_key
async def test_litellm_judge_scores_a_reply_with_a_real_model_call() -> None:
    judge = LiteLLMJudge()

    result = await judge.score(
        "What is the cancellation policy?",
        "This document does not cover that question.",
    )

    assert isinstance(result, QualityScore)
    assert 1 <= result.clarity <= 5
    assert 1 <= result.helpfulness <= 5
    assert result.rationale
