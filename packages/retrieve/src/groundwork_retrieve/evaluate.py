"""The retrieval evaluation harness: recall@3, recall@5, precision@5, and
MRR (section 8 of the day 5 prompt) computed against a set of eval
questions, for one chunking strategy and reranking setting at a time, then
across every one of this build's four real configuration combinations at
once. Build order step 20, still to come, is what points this at the real
45 question evalset/questions.yaml and persists EvalRun rows; this module
is the tested, reusable computation those later steps call, matching the
build order's own separation between step 11 (this harness) and step 20
(running it for real).

out_of_scope questions carry no expected_chunk_ids by construction (there
is no right answer chunk to rank against), so recall, precision, and MRR
are undefined for them rather than zero. evaluate_configuration excludes
that category entirely rather than letting an undefined case silently
drag every average toward zero: it is scored separately and
deterministically by checking the refusal behavior itself, build order
step 17, not by this harness. injection questions are the opposite case:
each one's single expected chunk is the planted hidden-instruction chunk,
and confirming it is actually retrieved, so the generation stage genuinely
had the chance to leak it, is exactly the signal section 11's defense test
needs, so that category stays in.

A question's expected_chunk_ids may legitimately name chunks from both
chunking strategies at once: scripts/build_eval_questions.py resolves
every grounding substring against both naive and structure aware chunks
and keeps the union, relying on search_chunks' own strategy scoping to
make that safe (a naive strategy search can never structurally return a
structure aware chunk, or the reverse). Scoring here re-derives, from the
database, which of a question's expected ids actually belong to the
strategy being scored right now, rather than trusting any assumed
ordering or split of that list. Skipping this step would silently cap
recall below 1.0 even for a perfect retrieval, since a naive search could
never reach a structure only id baked into the same expected_chunk_ids
list.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from groundwork_api.models import Chunk, EvalCategory
from groundwork_api.models import ChunkStrategy as DbChunkStrategy
from groundwork_core.model import StrictModel
from groundwork_retrieve.embeddings import Embedder, get_embedder
from groundwork_retrieve.rerank import Reranker, get_reranker
from groundwork_retrieve.search import search_chunks

DEFAULT_CANDIDATE_K = 20
DEFAULT_FINAL_K = 5

EXCLUDED_CATEGORIES = frozenset({EvalCategory.OUT_OF_SCOPE})
"""out_of_scope questions have no right-answer chunk by design (see module
docstring): excluded here, scored separately by the deterministic
refusal-checking step build order step 17 adds."""

CONFIGURATIONS: tuple[tuple[DbChunkStrategy, bool], ...] = (
    (DbChunkStrategy.NAIVE, False),
    (DbChunkStrategy.NAIVE, True),
    (DbChunkStrategy.STRUCTURE, False),
    (DbChunkStrategy.STRUCTURE, True),
)
"""Section 8's real configuration combinations in this build: both
chunking strategies, with and without reranking. The local versus API
embeddings axis section 8 also asks for is not a fifth and sixth entry
here, since no embedding API key exists in this sandbox: get_embedder()
always resolves to whichever one backend this environment actually has,
local model or the deterministic tfidf fallback, so there is no second
embedding backend to hold constant while comparing. RESULTS.md reports
that honestly as a gap rather than inventing a comparison cell that was
never run.
"""


class EvalQuestionRecord(StrictModel):
    """One scoreable question, normalized to the shape this module needs
    regardless of where it came from: the real evalset/questions.yaml by
    way of a later loader, or a handful of synthetic cases built directly
    in a test."""

    workspace_id: UUID
    question: str
    expected_chunk_ids: list[UUID]
    category: EvalCategory


class QuestionScore(StrictModel):
    """One question's metrics under one (strategy, reranker) configuration."""

    question: str
    category: EvalCategory
    recall_at_3: float
    recall_at_5: float
    precision_at_5: float
    reciprocal_rank: float


class MetricResult(StrictModel):
    """One row of the section 8 comparison table: one configuration,
    aggregated either across every scored question ("all") or scoped to
    one category, matching EvalRun.category's own documented dual
    purpose."""

    config_label: str
    category: str
    recall_at_3: float
    recall_at_5: float
    precision_at_5: float
    mrr: float
    n_questions: int


async def _scoped_expected_ids(
    session: AsyncSession, question: EvalQuestionRecord, strategy: DbChunkStrategy
) -> set[UUID]:
    """The subset of question.expected_chunk_ids that actually belongs to
    strategy, looked up against the database rather than assumed from the
    list's ordering. See the module docstring for why that matters."""
    if not question.expected_chunk_ids:
        return set()
    result = await session.execute(
        select(col(Chunk.id)).where(
            col(Chunk.id).in_(question.expected_chunk_ids),
            col(Chunk.workspace_id) == question.workspace_id,
            col(Chunk.strategy) == strategy,
        )
    )
    return set(result.scalars().all())


async def score_question(
    session: AsyncSession,
    *,
    question: EvalQuestionRecord,
    strategy: DbChunkStrategy,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    final_k: int = DEFAULT_FINAL_K,
) -> QuestionScore:
    """Scores one question under one (strategy, reranker) configuration.

    reranker=None scores the plain vector search order. Passing a
    Reranker widens the initial retrieval to candidate_k, reranks the
    whole candidate set, and scores the top final_k of that reordering
    instead, matching section 8's "retrieve a wider candidate set, then
    rerank... down to the final k".

    recall@3, recall@5, and MRR are all computed over the same final_k
    ranked list, whatever a real request would actually return to a
    caller, rather than a separately widened window a user would never
    see.

    Raises ValueError if strategy has no expected chunk at all for this
    question: out_of_scope questions must be filtered out before calling
    this (evaluate_configuration does that), and every other category is
    expected, by construction of evalset/questions.yaml, to resolve at
    least one chunk per strategy. An empty result here means that
    assumption broke, and failing loud beats silently scoring a 0.0 that
    looks like a real miss rather than a data bug.
    """
    scoped_expected = await _scoped_expected_ids(session, question, strategy)
    if not scoped_expected:
        raise ValueError(
            f"question {question.question!r} has no {strategy.value} expected chunk ids to "
            "score against; out_of_scope questions must be filtered out before calling "
            "score_question, and every other category must resolve at least one chunk "
            "per strategy"
        )

    embedder = embedder or get_embedder()
    if reranker is not None:
        candidates = await search_chunks(
            session,
            workspace_id=question.workspace_id,
            strategy=strategy,
            query=question.question,
            top_k=candidate_k,
            embedder=embedder,
        )
        scored_candidates = list(
            zip(
                candidates,
                reranker.rerank(question.question, [c.text for c in candidates]),
                strict=True,
            )
        )
        scored_candidates.sort(key=lambda pair: pair[1], reverse=True)
        ranked = [chunk for chunk, _ in scored_candidates][:final_k]
    else:
        ranked = await search_chunks(
            session,
            workspace_id=question.workspace_id,
            strategy=strategy,
            query=question.question,
            top_k=final_k,
            embedder=embedder,
        )

    ranked_ids = [chunk.id for chunk in ranked]
    hits_at_3 = len(scoped_expected.intersection(ranked_ids[:3]))
    hits_at_final_k = len(scoped_expected.intersection(ranked_ids[:final_k]))

    # Precision divides by how many chunks were actually considered, not a
    # fixed final_k: the injection red team workspace holds only a couple
    # of chunks in total, and precision@5 there should reward finding the
    # one correct chunk with 1.0, not cap it at 0.2 for a corpus too small
    # to ever fill five slots.
    precision_at_5 = hits_at_final_k / len(ranked_ids) if ranked_ids else 0.0

    reciprocal_rank = 0.0
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in scoped_expected:
            reciprocal_rank = 1.0 / rank
            break

    return QuestionScore(
        question=question.question,
        category=question.category,
        recall_at_3=hits_at_3 / len(scoped_expected),
        recall_at_5=hits_at_final_k / len(scoped_expected),
        precision_at_5=precision_at_5,
        reciprocal_rank=reciprocal_rank,
    )


def _config_label(strategy: DbChunkStrategy, use_reranking: bool) -> str:
    return f"{strategy.value}+{'rerank' if use_reranking else 'no_rerank'}"


def _aggregate(scores: list[QuestionScore], *, config_label: str, category: str) -> MetricResult:
    n = len(scores)
    return MetricResult(
        config_label=config_label,
        category=category,
        recall_at_3=sum(s.recall_at_3 for s in scores) / n,
        recall_at_5=sum(s.recall_at_5 for s in scores) / n,
        precision_at_5=sum(s.precision_at_5 for s in scores) / n,
        mrr=sum(s.reciprocal_rank for s in scores) / n,
        n_questions=n,
    )


async def evaluate_configuration(
    session: AsyncSession,
    *,
    questions: list[EvalQuestionRecord],
    strategy: DbChunkStrategy,
    config_label: str,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    final_k: int = DEFAULT_FINAL_K,
) -> list[MetricResult]:
    """Scores every question whose category is not excluded (see module
    docstring) one at a time. A plain sequential loop, not concurrent:
    AsyncSession is not safe to share across coroutines running at once.

    Returns one aggregate MetricResult across every scored question,
    labeled category="all", plus one further MetricResult per category
    actually present among them, matching EvalRun.category's documented
    dual purpose. Returns an empty list if every question passed in
    belongs to an excluded category.
    """
    embedder = embedder or get_embedder()
    scored: list[QuestionScore] = []
    for question in questions:
        if question.category in EXCLUDED_CATEGORIES:
            continue
        scored.append(
            await score_question(
                session,
                question=question,
                strategy=strategy,
                embedder=embedder,
                reranker=reranker,
                candidate_k=candidate_k,
                final_k=final_k,
            )
        )

    if not scored:
        return []

    results = [_aggregate(scored, config_label=config_label, category="all")]

    by_category: dict[EvalCategory, list[QuestionScore]] = {}
    for question_score in scored:
        by_category.setdefault(question_score.category, []).append(question_score)
    for category, group in by_category.items():
        results.append(_aggregate(group, config_label=config_label, category=category.value))

    return results


async def evaluate_all_configurations(
    session: AsyncSession,
    *,
    questions: list[EvalQuestionRecord],
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    final_k: int = DEFAULT_FINAL_K,
) -> list[MetricResult]:
    """Runs every one of CONFIGURATIONS against the same question set:
    both chunking strategies, with and without reranking, the four real
    combinations in this build (see CONFIGURATIONS' own docstring for why
    that is four and not six).

    reranker=None means "use the process wide get_reranker() for the
    rerank half of each pair", not "never rerank": the no_rerank half of
    every pair always scores with reranker explicitly withheld regardless
    of this argument, so a caller can never accidentally produce two
    identical configurations by forgetting to vary it. Pass an explicit
    reranker to control exactly what the rerank half uses, tests do this
    to stay deterministic.
    """
    embedder = embedder or get_embedder()
    results: list[MetricResult] = []
    for strategy, use_reranking in CONFIGURATIONS:
        active_reranker = (reranker or get_reranker()) if use_reranking else None
        results.extend(
            await evaluate_configuration(
                session,
                questions=questions,
                strategy=strategy,
                config_label=_config_label(strategy, use_reranking),
                embedder=embedder,
                reranker=active_reranker,
                candidate_k=candidate_k,
                final_k=final_k,
            )
        )
    return results
