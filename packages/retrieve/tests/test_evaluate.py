"""Retrieval evaluation harness tests. Per section 15 of the day 5 prompt:
metrics must be "computed and asserted non trivial (recall above zero, not
just that the function runs) against a small synthetic eval case with a
known right answer" before the real 45 question evalset/questions.yaml run
is trusted. Every test here builds that kind of small, fully controlled
case: a stub embedder pins down an exact, known similarity ranking (the
same technique test_search.py uses), so a "did it run" check (the right
number of results, the reranker actually invoked) and a "did it produce
the right numbers" check (recall, precision, and MRR matching a value
computed by hand from that known ranking) are both exercised, separately,
rather than folded into one loose assertion.
"""

from __future__ import annotations

from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from groundwork_api.models import (
    Chunk,
    ChunkStrategy,
    Document,
    EvalCategory,
    ExtractionMethod,
    Workspace,
)
from groundwork_core.ids import new_id
from groundwork_retrieve.evaluate import (
    CONFIGURATIONS,
    EvalQuestionRecord,
    evaluate_all_configurations,
    evaluate_configuration,
    score_question,
)

pytestmark = pytest.mark.requires_postgres


class _StubEmbedder:
    """Returns one fixed, caller supplied vector for any query text, same
    as test_search.py's own stub: what a query "means" is entirely pinned
    down by the test rather than by any real model."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]


def _vec(first: float, second: float) -> list[float]:
    v = [0.0] * 384
    v[0] = first
    v[1] = second
    return v


QUERY_VECTOR = _vec(1.0, 0.0)

# Six vectors at increasing angle from QUERY_VECTOR, so a cosine distance
# search against QUERY_VECTOR ranks them c1 (nearest) through c6 (a full
# reversal, farthest) in that exact, known order every time.
RANK_VECTORS = [
    _vec(1.0, 0.0),
    _vec(0.9, 0.1),
    _vec(0.8, 0.2),
    _vec(0.7, 0.3),
    _vec(0.6, 0.4),
    _vec(-1.0, 0.0),
]


class _RecordingReranker:
    """Reverses whatever order it is given (the weakest cosine match
    becomes the top rerank score) while recording every call, so a test
    can assert both that reranking actually ran for a configuration
    (calls is non-empty) and that it measurably changed the outcome
    relative to plain vector search order.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        self.calls.append((query, len(candidates)))
        return [float(i) for i in range(len(candidates))]


async def _make_workspace(session: AsyncSession, name: str) -> Workspace:
    workspace = Workspace(name=name)
    session.add(workspace)
    await session.flush()
    return workspace


async def _make_document(session: AsyncSession, workspace: Workspace, filename: str) -> Document:
    document = Document(
        workspace_id=workspace.id,
        filename=filename,
        sha256=new_id().hex * 2,  # a cheap, unique 64 character stand-in
        page_count=1,
        extraction_method=ExtractionMethod.TEXT,
    )
    session.add(document)
    await session.flush()
    return document


async def _seed_ranked_chunks(
    session: AsyncSession,
    *,
    workspace: Workspace,
    document: Document,
    strategy: ChunkStrategy,
    prefix: str,
) -> list[Chunk]:
    """Six chunks under strategy, ranked against QUERY_VECTOR in the exact
    known order c1 (nearest) through c6 (farthest) described by
    RANK_VECTORS, so every test below has a known right answer to check
    computed metrics against by hand.
    """
    chunks = [
        Chunk(
            document_id=document.id,
            workspace_id=workspace.id,
            strategy=strategy,
            text=f"{prefix}-c{i + 1}",
            page_start=1,
            page_end=1,
            char_start=0,
            char_end=len(f"{prefix}-c{i + 1}"),
            embedding=vector,
            embedding_backend="tfidf",
        )
        for i, vector in enumerate(RANK_VECTORS)
    ]
    for chunk in chunks:
        session.add(chunk)
    await session.flush()
    return chunks


async def _seed_dual_strategy_workspace(
    session: AsyncSession,
) -> tuple[Workspace, list[Chunk], list[Chunk]]:
    workspace = await _make_workspace(session, "eval-harness-test")
    document = await _make_document(session, workspace, "doc.pdf")
    naive_chunks = await _seed_ranked_chunks(
        session,
        workspace=workspace,
        document=document,
        strategy=ChunkStrategy.NAIVE,
        prefix="naive",
    )
    structure_chunks = await _seed_ranked_chunks(
        session,
        workspace=workspace,
        document=document,
        strategy=ChunkStrategy.STRUCTURE,
        prefix="structure",
    )
    await session.commit()
    return workspace, naive_chunks, structure_chunks


def _question(
    workspace_id: UUID,
    expected: list[UUID],
    category: EvalCategory = EvalCategory.DIRECT,
    text: str = "a test question",
) -> EvalQuestionRecord:
    return EvalQuestionRecord(
        workspace_id=workspace_id, question=text, expected_chunk_ids=expected, category=category
    )


# -- score_question: known-answer correctness --------------------------------


async def test_score_question_perfect_top_rank_scores_every_metric_at_its_maximum(
    db_session: AsyncSession,
) -> None:
    workspace, naive_chunks, _ = await _seed_dual_strategy_workspace(db_session)
    question = _question(workspace.id, [naive_chunks[0].id])

    score = await score_question(
        db_session,
        question=question,
        strategy=ChunkStrategy.NAIVE,
        embedder=_StubEmbedder(QUERY_VECTOR),
    )

    assert score.recall_at_3 == 1.0
    assert score.recall_at_5 == 1.0
    assert score.reciprocal_rank == 1.0
    assert score.precision_at_5 > 0.0


async def test_score_question_expected_chunk_outside_top_k_scores_every_metric_at_zero(
    db_session: AsyncSession,
) -> None:
    """The known miss case: c6 is the farthest chunk, ranked sixth of six,
    outside the default final_k=5 window, so this is a real miss rather
    than the function merely running.
    """
    workspace, naive_chunks, _ = await _seed_dual_strategy_workspace(db_session)
    question = _question(workspace.id, [naive_chunks[5].id])

    score = await score_question(
        db_session,
        question=question,
        strategy=ChunkStrategy.NAIVE,
        embedder=_StubEmbedder(QUERY_VECTOR),
    )

    assert score.recall_at_3 == 0.0
    assert score.recall_at_5 == 0.0
    assert score.reciprocal_rank == 0.0


async def test_score_question_partial_hit_across_two_expected_chunks_gives_proportional_recall(
    db_session: AsyncSession,
) -> None:
    """The boundary spanning shape: two expected chunks, one found in the
    top k and one not. Recall is the fraction found, not an all-or-nothing
    indicator, since a two-chunk answer half retrieved is a real partial
    result and section 7's chunking comparison depends on that gradation
    to be visible rather than rounded away.
    """
    workspace, naive_chunks, _ = await _seed_dual_strategy_workspace(db_session)
    question = _question(workspace.id, [naive_chunks[0].id, naive_chunks[5].id])

    score = await score_question(
        db_session,
        question=question,
        strategy=ChunkStrategy.NAIVE,
        embedder=_StubEmbedder(QUERY_VECTOR),
    )

    assert score.recall_at_5 == 0.5
    # The found half is ranked first, so MRR still reports a perfect
    # reciprocal rank: MRR asks how far down the first hit is, not
    # whether every expected chunk was found.
    assert score.reciprocal_rank == 1.0


async def test_score_question_reciprocal_rank_reflects_actual_rank_position(
    db_session: AsyncSession,
) -> None:
    workspace, naive_chunks, _ = await _seed_dual_strategy_workspace(db_session)
    question = _question(workspace.id, [naive_chunks[2].id])  # third nearest, known rank 3

    score = await score_question(
        db_session,
        question=question,
        strategy=ChunkStrategy.NAIVE,
        embedder=_StubEmbedder(QUERY_VECTOR),
    )

    assert score.reciprocal_rank == pytest.approx(1 / 3)
    assert score.recall_at_3 == 1.0


async def test_score_question_precision_denominator_is_the_actual_corpus_size_not_a_fixed_five(
    db_session: AsyncSession,
) -> None:
    """A workspace with only one chunk in total can still score a perfect
    precision@5 of 1.0: penalizing it down to 1/5 purely because the
    corpus is smaller than k would make a tiny demo workspace (the
    injection red team fixture is exactly this small in the real corpus)
    structurally unable to ever reach perfect precision.
    """
    workspace = await _make_workspace(db_session, "single-chunk-workspace")
    document = await _make_document(db_session, workspace, "doc.pdf")
    only_chunk = Chunk(
        document_id=document.id,
        workspace_id=workspace.id,
        strategy=ChunkStrategy.NAIVE,
        text="the only chunk",
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=15,
        embedding=QUERY_VECTOR,
        embedding_backend="tfidf",
    )
    db_session.add(only_chunk)
    await db_session.commit()

    question = _question(workspace.id, [only_chunk.id])
    score = await score_question(
        db_session,
        question=question,
        strategy=ChunkStrategy.NAIVE,
        embedder=_StubEmbedder(QUERY_VECTOR),
    )

    assert score.precision_at_5 == 1.0
    assert score.recall_at_5 == 1.0


async def test_score_question_raises_when_no_expected_chunk_resolves_for_the_strategy(
    db_session: AsyncSession,
) -> None:
    """The refuses-on-nothing case: an out_of_scope-shaped question (no
    expected chunks at all) must never silently score as a 0.0 that looks
    like a real miss. score_question fails loudly instead, so a caller
    that forgot to filter out_of_scope questions finds out immediately.
    """
    workspace, _, _ = await _seed_dual_strategy_workspace(db_session)
    question = _question(workspace.id, [], category=EvalCategory.OUT_OF_SCOPE)

    with pytest.raises(ValueError, match="no naive expected chunk ids"):
        await score_question(
            db_session,
            question=question,
            strategy=ChunkStrategy.NAIVE,
            embedder=_StubEmbedder(QUERY_VECTOR),
        )


# -- strategy scoping of mixed expected_chunk_ids -----------------------------


async def test_score_question_scopes_mixed_strategy_expected_ids_to_the_requested_strategy(
    db_session: AsyncSession,
) -> None:
    """The real evalset/questions.yaml shape: one question's
    expected_chunk_ids can legitimately mix a naive chunk id and a
    structure aware chunk id in the same flat list (see evaluate.py's
    module docstring). Scoring under one strategy must count only the
    matching half. If the denominator were the raw, unscoped
    expected_chunk_ids length (two) instead of the strategy-scoped subset
    (one), a perfect single-strategy retrieval would wrongly read back as
    0.5 recall instead of 1.0, exactly what this test would catch.
    """
    workspace, naive_chunks, structure_chunks = await _seed_dual_strategy_workspace(db_session)
    mixed_expected = [naive_chunks[0].id, structure_chunks[0].id]

    naive_score = await score_question(
        db_session,
        question=_question(workspace.id, mixed_expected),
        strategy=ChunkStrategy.NAIVE,
        embedder=_StubEmbedder(QUERY_VECTOR),
    )
    structure_score = await score_question(
        db_session,
        question=_question(workspace.id, mixed_expected),
        strategy=ChunkStrategy.STRUCTURE,
        embedder=_StubEmbedder(QUERY_VECTOR),
    )

    assert naive_score.recall_at_5 == 1.0
    assert structure_score.recall_at_5 == 1.0


# -- score_question: reranking actually changes the ranking used -------------


async def test_score_question_with_a_reranker_scores_the_reranked_order_not_the_vector_order(
    db_session: AsyncSession,
) -> None:
    """_RecordingReranker reverses the candidate order it is given, so the
    vector search's farthest match (c6, a known miss with no reranker)
    becomes the top ranked result once reranking runs, a real miss turned
    into a real hit rather than a coincidence.
    """
    workspace, naive_chunks, _ = await _seed_dual_strategy_workspace(db_session)
    question = _question(workspace.id, [naive_chunks[5].id])
    reranker = _RecordingReranker()

    score = await score_question(
        db_session,
        question=question,
        strategy=ChunkStrategy.NAIVE,
        embedder=_StubEmbedder(QUERY_VECTOR),
        reranker=reranker,
        candidate_k=6,
    )

    assert reranker.calls == [("a test question", 6)]
    assert score.reciprocal_rank == 1.0


# -- evaluate_configuration: aggregation and category exclusion --------------


async def test_evaluate_configuration_excludes_out_of_scope_and_aggregates_the_rest(
    db_session: AsyncSession,
) -> None:
    workspace, naive_chunks, _ = await _seed_dual_strategy_workspace(db_session)
    questions = [
        _question(
            workspace.id, [naive_chunks[0].id], category=EvalCategory.DIRECT, text="direct one"
        ),
        _question(
            workspace.id, [naive_chunks[0].id], category=EvalCategory.DIRECT, text="direct two"
        ),
        _question(
            workspace.id, [naive_chunks[5].id], category=EvalCategory.BOUNDARY, text="boundary one"
        ),
        _question(workspace.id, [], category=EvalCategory.OUT_OF_SCOPE, text="out of scope one"),
    ]

    results = await evaluate_configuration(
        db_session,
        questions=questions,
        strategy=ChunkStrategy.NAIVE,
        config_label="naive+no_rerank",
        embedder=_StubEmbedder(QUERY_VECTOR),
    )

    by_category = {result.category: result for result in results}
    assert set(by_category) == {"all", "direct", "boundary"}
    # The out_of_scope question contributes to no row at all, including "all".
    assert by_category["all"].n_questions == 3
    assert by_category["direct"].n_questions == 2
    assert by_category["direct"].recall_at_5 == 1.0
    assert by_category["boundary"].n_questions == 1
    assert by_category["boundary"].recall_at_5 == 0.0
    # "all" is the mean over every scored question, both hits and the one miss.
    assert by_category["all"].recall_at_5 == pytest.approx((1.0 + 1.0 + 0.0) / 3)


async def test_evaluate_configuration_returns_empty_list_when_every_question_is_excluded(
    db_session: AsyncSession,
) -> None:
    workspace, _, _ = await _seed_dual_strategy_workspace(db_session)
    questions = [_question(workspace.id, [], category=EvalCategory.OUT_OF_SCOPE)]

    results = await evaluate_configuration(
        db_session,
        questions=questions,
        strategy=ChunkStrategy.NAIVE,
        config_label="naive+no_rerank",
        embedder=_StubEmbedder(QUERY_VECTOR),
    )

    assert results == []


# -- evaluate_all_configurations: every real combination, reranker gated ----


async def test_evaluate_all_configurations_covers_all_four_combinations_and_reranks_selectively(
    db_session: AsyncSession,
) -> None:
    workspace, naive_chunks, structure_chunks = await _seed_dual_strategy_workspace(db_session)
    mixed_expected = [naive_chunks[0].id, structure_chunks[0].id]
    questions = [_question(workspace.id, mixed_expected, category=EvalCategory.BOUNDARY)]
    reranker = _RecordingReranker()

    results = await evaluate_all_configurations(
        db_session,
        questions=questions,
        embedder=_StubEmbedder(QUERY_VECTOR),
        reranker=reranker,
        candidate_k=6,
    )

    assert {result.config_label for result in results} == {
        "naive+no_rerank",
        "naive+rerank",
        "structure+no_rerank",
        "structure+rerank",
    }
    # Exactly one call per rerank configuration (naive+rerank,
    # structure+rerank); the two no_rerank configurations must never
    # reach the reranker at all.
    assert len(reranker.calls) == 2


def test_configurations_is_exactly_the_four_real_combinations_this_build_has() -> None:
    """Documents, and guards, the section 8 axis this build actually has:
    both chunking strategies crossed with reranking on or off. The local
    versus API embeddings axis is not a third dimension here, since this
    sandbox has only ever resolved one embedding backend (see
    evaluate.py's module docstring); a future environment that adds a
    real embedding API key would extend this tuple, not this test's
    expectations of it today.
    """
    assert len(CONFIGURATIONS) == 4
    assert len(set(CONFIGURATIONS)) == 4
