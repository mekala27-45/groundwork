"""groundwork_api.chat.ask tests: the full retrieve, generate, verify,
persist path, against a real pgvector backed database. Workspace boundary
specific behavior lives in test_isolation.py; this file covers ask()'s own
correctness, independent of any cross-workspace concern.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from groundwork_api.chat import ConversationNotFoundError, ask
from groundwork_api.models import (
    Chunk,
    ChunkStrategy,
    Conversation,
    Document,
    ExtractionMethod,
    Workspace,
)
from groundwork_core.ids import new_id
from groundwork_generate.generate import ExtractiveGenerator, GenerationResult
from groundwork_verify.quality import QualityScore

pytestmark = pytest.mark.requires_postgres


class _StubEmbedder:
    """Returns one fixed vector for any text, so a test controls exactly
    how similar a query is judged to be to a stored chunk, matching the
    convention already established in test_search.py and test_evaluate.py."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]


def _vec(first: float, second: float) -> list[float]:
    v = [0.0] * 384
    v[0] = first
    v[1] = second
    return v


class _RecordingGenerator:
    """Defers to a real ExtractiveGenerator for its answer, while
    recording the chunks it actually received, so a test can check "did
    the right input reach generation" separately from "did generation
    produce the right output", the carried forward rule this whole build
    applies to every gate."""

    def __init__(self) -> None:
        self.received_chunks: list[Chunk] | None = None
        self._inner = ExtractiveGenerator()

    async def generate(self, question: str, chunks: list[Chunk]) -> GenerationResult:
        self.received_chunks = chunks
        return await self._inner.generate(question, chunks)


class _StubJudge:
    """Returns a fixed QualityScore with no network access, the same
    recording-stub shape _RecordingGenerator uses, so ask()'s own wiring
    (does the answer reach the judge, does the result land in
    Turn.judge_scores) is provable independent of groundwork_verify's own
    LiteLLMJudge tests."""

    def __init__(self, result: QualityScore | None) -> None:
        self._result = result

    async def score(self, question: str, answer: str) -> QualityScore | None:
        return self._result


class _RecordingReranker:
    """Reverses whatever order it is given (increasing scores by index, so
    the weakest vector match becomes the top rerank score) while recording
    every call, the same convention test_evaluate.py's own
    _RecordingReranker uses, so a test can confirm reranking actually ran
    and measurably changed the outcome, not merely that it was called."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        self.calls.append((query, candidates))
        return [float(i) for i in range(len(candidates))]


async def _seed_workspace_with_chunk(
    session: AsyncSession, *, text: str = "some relevant text", embedding: list[float] | None = None
) -> tuple[Workspace, Chunk]:
    workspace = Workspace(name="chat-test")
    session.add(workspace)
    await session.flush()

    document = Document(
        workspace_id=workspace.id,
        filename="doc.pdf",
        sha256=new_id().hex * 2,
        page_count=1,
        extraction_method=ExtractionMethod.TEXT,
    )
    session.add(document)
    await session.flush()

    chunk = Chunk(
        document_id=document.id,
        workspace_id=workspace.id,
        strategy=ChunkStrategy.NAIVE,
        text=text,
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(text),
        embedding=embedding or _vec(1.0, 0.0),
        embedding_backend="tfidf",
    )
    session.add(chunk)
    await session.commit()
    return workspace, chunk


async def _seed_conversation(session: AsyncSession, workspace: Workspace) -> Conversation:
    conversation = Conversation(workspace_id=workspace.id)
    session.add(conversation)
    await session.commit()
    return conversation


async def test_ask_persists_a_turn_with_the_generated_answer_and_verified_citation(
    db_session: AsyncSession,
) -> None:
    workspace, chunk = await _seed_workspace_with_chunk(
        db_session, text="Sessions run fifty minutes."
    )
    conversation = await _seed_conversation(db_session, workspace)

    turn = await ask(
        db_session,
        conversation_id=conversation.id,
        question="How long is a session?",
        strategy=ChunkStrategy.NAIVE,
        embedder=_StubEmbedder(_vec(1.0, 0.0)),
        generator=ExtractiveGenerator(),
    )

    assert turn.workspace_id == workspace.id
    assert turn.conversation_id == conversation.id
    assert turn.extractive_fallback is True
    assert "Sessions run fifty minutes." in turn.answer
    assert turn.retrieved_chunk_ids == [str(chunk.id)]
    assert len(turn.citation_verifications) == 1
    assert turn.citation_verifications[0]["chunk_id"] == str(chunk.id)
    assert turn.citation_verifications[0]["verified"] is True
    assert len(turn.claims) == 1
    assert turn.claims[0]["nli_label"] == "entailed"


async def test_ask_raises_when_the_conversation_does_not_exist(db_session: AsyncSession) -> None:
    with pytest.raises(ConversationNotFoundError):
        await ask(
            db_session,
            conversation_id=new_id(),
            question="anything",
            strategy=ChunkStrategy.NAIVE,
            embedder=_StubEmbedder(_vec(1.0, 0.0)),
        )


async def test_ask_answers_not_covered_when_nothing_is_retrieved_at_all(
    db_session: AsyncSession,
) -> None:
    workspace = Workspace(name="empty-workspace")
    db_session.add(workspace)
    await db_session.flush()
    conversation = await _seed_conversation(db_session, workspace)

    turn = await ask(
        db_session,
        conversation_id=conversation.id,
        question="anything",
        strategy=ChunkStrategy.NAIVE,
        embedder=_StubEmbedder(_vec(1.0, 0.0)),
        generator=ExtractiveGenerator(),
    )

    assert turn.answer == "This document does not cover that question."
    assert turn.retrieved_chunk_ids == []
    assert turn.citation_verifications == []


async def test_ask_refuses_when_the_top_chunk_is_below_the_relevance_threshold(
    db_session: AsyncSession,
) -> None:
    """The relevance gate this module adds: retrieval found something, but
    it is orthogonal (cosine similarity 0.0) to the query, well below the
    default threshold, so generation must never see it.
    """
    workspace, chunk = await _seed_workspace_with_chunk(db_session, embedding=_vec(0.0, 1.0))
    conversation = await _seed_conversation(db_session, workspace)
    generator = _RecordingGenerator()

    turn = await ask(
        db_session,
        conversation_id=conversation.id,
        question="an unrelated query",
        strategy=ChunkStrategy.NAIVE,
        embedder=_StubEmbedder(_vec(1.0, 0.0)),
        generator=generator,
    )

    assert generator.received_chunks == []
    assert turn.answer == "This document does not cover that question."
    assert turn.citation_verifications == []
    # Retrieval itself still found the chunk; only generation was gated,
    # so the trace can show a reader what was found and why it was judged
    # irrelevant rather than looking identical to an empty corpus.
    assert turn.retrieved_chunk_ids == [str(chunk.id)]


async def test_ask_does_not_gate_a_clearly_relevant_top_chunk(db_session: AsyncSession) -> None:
    generator = _RecordingGenerator()
    workspace, chunk = await _seed_workspace_with_chunk(db_session, embedding=_vec(1.0, 0.0))
    conversation = await _seed_conversation(db_session, workspace)

    turn = await ask(
        db_session,
        conversation_id=conversation.id,
        question="a well matched query",
        strategy=ChunkStrategy.NAIVE,
        embedder=_StubEmbedder(_vec(1.0, 0.0)),
        generator=generator,
    )

    assert generator.received_chunks == [chunk]
    assert turn.extractive_fallback is True
    assert turn.answer != "This document does not cover that question."


async def test_ask_applies_reranking_when_requested_and_records_it(
    db_session: AsyncSession,
) -> None:
    workspace = Workspace(name="rerank-test")
    db_session.add(workspace)
    await db_session.flush()
    document = Document(
        workspace_id=workspace.id,
        filename="d.pdf",
        sha256=new_id().hex * 2,
        page_count=1,
        extraction_method=ExtractionMethod.TEXT,
    )
    db_session.add(document)
    await db_session.flush()
    first = Chunk(
        document_id=document.id,
        workspace_id=workspace.id,
        strategy=ChunkStrategy.NAIVE,
        text="ranked first by vector search",
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=1,
        embedding=_vec(1.0, 0.0),
        embedding_backend="tfidf",
    )
    second = Chunk(
        document_id=document.id,
        workspace_id=workspace.id,
        strategy=ChunkStrategy.NAIVE,
        text="ranked second by vector search but first after rerank",
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=1,
        embedding=_vec(0.9, 0.1),
        embedding_backend="tfidf",
    )
    db_session.add(first)
    db_session.add(second)
    await db_session.commit()
    conversation = await _seed_conversation(db_session, workspace)
    reranker = _RecordingReranker()
    generator = _RecordingGenerator()

    turn = await ask(
        db_session,
        conversation_id=conversation.id,
        question="q",
        strategy=ChunkStrategy.NAIVE,
        use_reranking=True,
        embedder=_StubEmbedder(_vec(1.0, 0.0)),
        reranker=reranker,
        generator=generator,
    )

    assert len(reranker.calls) == 1
    assert turn.reranked is True
    # _RecordingReranker reverses order: second (originally ranked below
    # first by vector distance) must come first after reranking.
    assert generator.received_chunks is not None
    assert generator.received_chunks[0].id == second.id


async def test_ask_uses_the_requested_chunking_strategy(db_session: AsyncSession) -> None:
    workspace = Workspace(name="strategy-test")
    db_session.add(workspace)
    await db_session.flush()
    document = Document(
        workspace_id=workspace.id,
        filename="d.pdf",
        sha256=new_id().hex * 2,
        page_count=1,
        extraction_method=ExtractionMethod.TEXT,
    )
    db_session.add(document)
    await db_session.flush()
    naive_chunk = Chunk(
        document_id=document.id,
        workspace_id=workspace.id,
        strategy=ChunkStrategy.NAIVE,
        text="naive chunk",
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=1,
        embedding=_vec(1.0, 0.0),
        embedding_backend="tfidf",
    )
    structure_chunk = Chunk(
        document_id=document.id,
        workspace_id=workspace.id,
        strategy=ChunkStrategy.STRUCTURE,
        text="structure chunk",
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=1,
        embedding=_vec(1.0, 0.0),
        embedding_backend="tfidf",
    )
    db_session.add(naive_chunk)
    db_session.add(structure_chunk)
    await db_session.commit()
    conversation = await _seed_conversation(db_session, workspace)

    turn = await ask(
        db_session,
        conversation_id=conversation.id,
        question="q",
        strategy=ChunkStrategy.STRUCTURE,
        embedder=_StubEmbedder(_vec(1.0, 0.0)),
        generator=ExtractiveGenerator(),
    )

    assert turn.retrieved_chunk_ids == [str(structure_chunk.id)]
    assert turn.chunking_strategy == ChunkStrategy.STRUCTURE


async def test_ask_writes_judge_scores_from_the_provided_judge(db_session: AsyncSession) -> None:
    """Turn.judge_scores comes from whatever Judge ask() is given, stored
    as the plain dict a real caller (the /trace page) can render directly,
    entirely separate from turn.claims and turn.citation_verifications,
    the faithfulness gate's own columns."""
    workspace, _ = await _seed_workspace_with_chunk(db_session, text="Sessions run fifty minutes.")
    conversation = await _seed_conversation(db_session, workspace)
    quality = QualityScore(clarity=4, helpfulness=5, rationale="Answers the question directly.")

    turn = await ask(
        db_session,
        conversation_id=conversation.id,
        question="How long is a session?",
        strategy=ChunkStrategy.NAIVE,
        embedder=_StubEmbedder(_vec(1.0, 0.0)),
        generator=ExtractiveGenerator(),
        judge=_StubJudge(quality),
    )

    assert turn.judge_scores == quality.model_dump()


async def test_ask_leaves_judge_scores_none_with_no_judge_and_no_key(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real default in this sandbox: nothing passed for judge, and no
    LLM key configured, so the secondary quality metric stays an honest
    None rather than a fabricated score."""
    monkeypatch.delenv("GROUNDWORK_LLM_API_KEY", raising=False)
    workspace, _ = await _seed_workspace_with_chunk(db_session, text="Sessions run fifty minutes.")
    conversation = await _seed_conversation(db_session, workspace)

    turn = await ask(
        db_session,
        conversation_id=conversation.id,
        question="How long is a session?",
        strategy=ChunkStrategy.NAIVE,
        embedder=_StubEmbedder(_vec(1.0, 0.0)),
        generator=ExtractiveGenerator(),
    )

    assert turn.judge_scores is None
