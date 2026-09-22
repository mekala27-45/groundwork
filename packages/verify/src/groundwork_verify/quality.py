"""Secondary quality scoring: reply clarity and reply helpfulness, never
faithfulness, section 10's own explicit requirement ("a LiteLLMJudge, same
forced structured output pattern as Day 1, scoring reply clarity and
helpfulness only, never faithfulness, and labeled in every output as a
secondary metric").

Kept structurally separate from groundwork_verify.faithfulness on purpose,
the judge conflict of interest lesson applied a second time in this build
(the first was groundwork_verify.faithfulness itself, an independent NLI
model rather than the generation model grading its own answer). A judge
model scoring how clearly a reply reads is a different kind of question
than whether the reply is true or grounded, and this module never lets the
two mix: QualityScore has no faithfulness field, groundwork_api.chat.ask
writes it to Turn.judge_scores, a column entirely separate from
Turn.claims and Turn.citation_verifications, and nothing in this build
computes a pass or fail gate from it. RESULTS.md and the eval dashboard
label every figure sourced from here a secondary metric, never a
substitute for the faithfulness scorecard next to it.

The forced structured output pattern, reused from Day 1 (trajectory,
described in claude/build-series-log.md as "forced tool-calling structured
output, one named field per judged dimension rather than a list, retry
with the validation error appended"): the judge is offered exactly one
tool, record_quality_score, its parameters schema generated directly from
QualityScore itself so the schema the model is shown can never drift from
the schema its answer is validated against, and tool_choice forces that
one call rather than hoping the model volunteers it. parse_quality_score
is kept as its own small, pure, network free function specifically so this
parse-then-validate step is unit tested directly, not only reachable
through a real, skip marked model call.

Unlike embeddings, reranking, and faithfulness scoring, this dimension
ships no offline, deterministic, open weight substitute in this build.
There is no local model standing in for a person's judgment of how
clearly a reply reads. Day 1 hit the identical wall: three of its ten
failure mode detectors needed a rubric judge, and with no key configured
its own build log states plainly "no model is measured." The same choice
is made again here rather than inventing a fallback that would just be
guessing with extra steps: with no LLM key configured, or an exhausted
spend ledger, score_reply_quality returns None, and Turn.judge_scores
stores that as a missing value, not a synthetic score. This sandbox has no
key, so every quality score in RESULTS.md is reported as unmeasured here,
exactly as the LiteLLM generation path itself is, rather than silently
filled in.
"""

from __future__ import annotations

import json
from typing import Literal, Protocol

from pydantic import Field, ValidationError

from groundwork_core.config import get_settings
from groundwork_core.model import StrictModel
from groundwork_generate.ledger import SpendLedger, get_ledger

QualityBackend = Literal["litellm_judge", "unavailable"]

RECORD_QUALITY_SCORE_TOOL = "record_quality_score"

JUDGE_SYSTEM_PROMPT = (
    "You are scoring the quality of a written reply, not its factual "
    "accuracy. Call record_quality_score exactly once with your judgment. "
    "Score clarity: how easy the reply is to read and understand on its "
    "own terms. Score helpfulness: how well the reply addresses what the "
    "person actually asked, including that a plainly stated refusal is "
    "genuinely helpful when the source material does not cover the "
    "question at all. Never judge whether the reply is true, grounded, or "
    "faithful to any source: that is scored elsewhere, by a different, "
    "independent method, and is not your job here."
)


class QualityScore(StrictModel):
    """Two named dimensions, not a list of {dimension, value} pairs, per
    Day 1's own pattern: forcing the model to fill two required, separately
    named, separately bounded fields leaves less room for a malformed or
    partial structured response than a generic list would."""

    clarity: int = Field(
        ge=1, le=5, description="How easy the reply is to read and understand, 1 low to 5 high."
    )
    helpfulness: int = Field(
        ge=1,
        le=5,
        description=(
            "How well the reply addresses what was actually asked, 1 low to 5 high. "
            "A stated refusal scores high here when the source material genuinely "
            "does not cover the question: refusing correctly is helpful."
        ),
    )
    rationale: str = Field(
        description="One or two sentences on why, naming something specific in the reply."
    )


def _tool_schema() -> dict[str, object]:
    parameters = QualityScore.model_json_schema()
    parameters.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": RECORD_QUALITY_SCORE_TOOL,
            "description": (
                "Record a secondary quality judgment of a RAG system's reply: how "
                "clearly it reads and how helpful it would be to the person who "
                "asked, each on a 1 to 5 scale. Never a judgment of whether the "
                "reply is factually correct or grounded in its sources."
            ),
            "parameters": parameters,
        },
    }


def _user_prompt(question: str, answer: str) -> str:
    return f"Question asked:\n{question}\n\nReply to score:\n{answer}"


def parse_quality_score(raw_arguments: str) -> QualityScore:
    """Turns one forced tool call's raw JSON arguments string into a
    validated QualityScore, raising ValueError, wrapping the underlying
    json.JSONDecodeError or pydantic.ValidationError, on anything that does
    not satisfy the schema. A pure function, no network and no litellm
    object shapes involved, so the parsing and validation this whole module
    exists to force is exercised directly by a normal unit test rather than
    only reachable through a real, skip marked model call.
    """
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(f"tool call arguments were not valid JSON: {exc}") from exc
    try:
        return QualityScore.model_validate(arguments)
    except ValidationError as exc:
        raise ValueError(f"tool call arguments did not match the schema: {exc}") from exc


class Judge(Protocol):
    async def score(self, question: str, answer: str) -> QualityScore | None:
        """None means no secondary score was produced: no key configured,
        the shared spend ledger already exhausted, or a judge that never
        returned usable arguments even after retrying once. Never a
        fabricated placeholder score standing in for a judgment nothing
        actually made."""
        ...


class LiteLLMJudge:
    """A real model call through LiteLLM, forced to answer through exactly
    one tool call rather than free text, with one bounded retry that
    appends the validation error to the conversation when the first call's
    arguments do not parse or do not satisfy QualityScore's schema.

    Shares the same SpendLedger every LiteLLMGenerator call draws down,
    checked before every attempt here exactly as it is before every
    generation call, so a judge run can never spend past the one ceiling
    a generation run already respects. Defaults to Settings.judge_model
    when one is configured, falling back to Settings.llm_model otherwise:
    an operator who wants the model being judged to never also be the
    model doing the judging, the same conflict of interest reasoning
    groundwork_verify.faithfulness applies structurally, can set
    judge_model to a different model without any code change; leaving it
    unset still gets a working judge, just not that additional guarantee.

    This class assumes a configured key, exactly as LiteLLMGenerator does:
    the decision of whether to construct one at all belongs to
    score_reply_quality below, not to this class checking its own
    settings. Unmeasured in this sandbox, which has no LLM API key; its
    tests are skip marked behind requires_llm_key, the same shape every
    other real model call in this build uses.
    """

    def __init__(
        self,
        model: str | None = None,
        ledger: SpendLedger | None = None,
        *,
        max_attempts: int = 2,
    ) -> None:
        settings = get_settings()
        self._model = model or settings.judge_model or settings.llm_model
        self._api_key = settings.llm_api_key
        self._ledger = ledger or get_ledger()
        self._max_attempts = max_attempts

    async def score(self, question: str, answer: str) -> QualityScore | None:
        if not self._ledger.has_budget():
            return None

        import litellm

        messages: list[dict[str, object]] = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(question, answer)},
        ]
        tools = [_tool_schema()]
        tool_choice = {"type": "function", "function": {"name": RECORD_QUALITY_SCORE_TOOL}}

        for attempt in range(self._max_attempts):
            if not self._ledger.has_budget():
                return None

            response = await litellm.acompletion(
                model=self._model,
                api_key=self._api_key,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )
            cost_usd = float(litellm.completion_cost(completion_response=response) or 0.0)
            self._ledger.record(cost_usd)

            message = response.choices[0].message
            tool_calls = message.tool_calls or []
            if not tool_calls:
                return None
            call = tool_calls[0]

            try:
                return parse_quality_score(call.function.arguments)
            except ValueError as exc:
                if attempt + 1 >= self._max_attempts:
                    return None
                messages.append(message.model_dump(exclude_none=True))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": (
                            f"Invalid: {exc}. Call {RECORD_QUALITY_SCORE_TOOL} again "
                            "with corrected arguments that satisfy the schema."
                        ),
                    }
                )

        return None


def resolve_quality_backend() -> QualityBackend:
    """Whether a real judge call is even possible right now: is a key
    configured at all, the identical local fact
    groundwork_generate.generate.resolve_generation_backend checks for the
    same reason. No probe, no local fallback: see this module's own
    docstring for why quality scoring has none to fall back to."""
    return "litellm_judge" if get_settings().llm_api_key else "unavailable"


async def score_reply_quality(
    question: str, answer: str, *, judge: Judge | None = None
) -> QualityScore | None:
    """The one entry point most callers should use, resolving which
    backend to construct fresh on every call, the same shape
    groundwork_generate.generate.generate_answer uses and for the same
    reason: a missing key is stable for the process, but the shared spend
    ledger's remaining budget is not. Pass an explicit judge to bypass
    resolution entirely, which tests do to stay fast, deterministic, and
    network free.
    """
    active_judge = judge or (
        LiteLLMJudge() if resolve_quality_backend() == "litellm_judge" else None
    )
    if active_judge is None:
        return None
    return await active_judge.score(question, answer)
