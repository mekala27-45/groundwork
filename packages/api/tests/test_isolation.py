"""test_workspace_isolation: the day 5 prompt's own required test (section
5, "seed two workspaces, attempt retrieval and chat across the boundary
through every code path, assert refusal every time, not silent
filtering"), the same shape as day 4's tenant isolation test applied to
document workspaces instead of clinics. models.py's own module docstring
already points here by name.

Three code paths, each attempted directly rather than trusted from a
neighbouring test suite's own coverage: groundwork_retrieve.search.search_chunks
(the raw retrieval query), groundwork_api.chat.ask (the full chat path this
build adds in this same commit), and groundwork_verify.citation.verify_citations
(deterministic citation checking). All three enforce the boundary the same
way: an explicit, workspace_id scoped query, or an explicit, named failure
reason, never a code path that merely happens to return nothing because it
forgot to look in the first place.

The test deliberately makes the other workspace's chunk the objectively
closer vector match to the query used against the workspace under test,
so a scoping bug would have every incentive to leak it, rather than
picking vectors that would pass even with the workspace_id filter quietly
missing.

One path this file does not attempt: asking ask() to produce a citation
to a chunk it never actually retrieved. Neither Generator implementation
can do that by construction (GenerationResult.cited_chunk_ids is always
built directly from the chunks list generate_answer() was handed), so
forcing it would mean testing a third, adversarial Generator that does
not exist anywhere in this system, not a code path this build actually
has. The realistic version of that same risk, a citation naming a real
chunk that belongs to a different workspace, is exactly what the
verify_citations checks below prove is caught, tested directly at that
function's own boundary rather than staged through an artificial
generator that would only obscure what is actually being proven.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from groundwork_api.chat import ask
from groundwork_api.models import (
    Chunk,
    ChunkStrategy,
    Conversation,
    Document,
    ExtractionMethod,
    Workspace,
)
from groundwork_core.ids import new_id
from groundwork_generate.generate import ExtractiveGenerator
from groundwork_retrieve.search import search_chunks
from groundwork_verify.citation import verify_citations

pytestmark = pytest.mark.requires_postgres


class _StubEmbedder:
    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]


def _vec(first: float, second: float) -> list[float]:
    v = [0.0] * 384
    v[0] = first
    v[1] = second
    return v


async def _seed_workspace(
    session: AsyncSession, *, name: str, chunk_text: str, embedding: list[float]
) -> tuple[Workspace, Chunk]:
    workspace = Workspace(name=name)
    session.add(workspace)
    await session.flush()
    document = Document(
        workspace_id=workspace.id,
        filename="d.pdf",
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
        text=chunk_text,
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(chunk_text),
        embedding=embedding,
        embedding_backend="tfidf",
    )
    session.add(chunk)
    await session.commit()
    return workspace, chunk


async def test_workspace_isolation(db_session: AsyncSession) -> None:
    query_vector = _vec(1.0, 0.0)
    workspace_a, chunk_a = await _seed_workspace(
        db_session,
        name="workspace-a",
        chunk_text="workspace a content, the only thing workspace a's own conversation may see",
        embedding=_vec(0.6, 0.8),
    )
    workspace_b, chunk_b = await _seed_workspace(
        db_session,
        name="workspace-b",
        chunk_text="workspace b content, must never leak into workspace a's answers",
        embedding=_vec(1.0, 0.0),
    )
    embedder = _StubEmbedder(query_vector)

    # Path 1: the raw retrieval query. chunk_b is the objectively closer
    # vector match (cosine similarity 1.0 against chunk_a's 0.6), so a
    # scoping bug has every incentive to return it here.
    results_a = await search_chunks(
        db_session,
        workspace_id=workspace_a.id,
        strategy=ChunkStrategy.NAIVE,
        query="q",
        embedder=embedder,
    )
    assert [r.id for r in results_a] == [chunk_a.id]

    # Path 2: the full chat path.
    conversation_a = Conversation(workspace_id=workspace_a.id)
    db_session.add(conversation_a)
    await db_session.commit()

    turn = await ask(
        db_session,
        conversation_id=conversation_a.id,
        question="q",
        strategy=ChunkStrategy.NAIVE,
        embedder=embedder,
        generator=ExtractiveGenerator(),
    )
    assert turn.workspace_id == workspace_a.id
    assert turn.retrieved_chunk_ids == [str(chunk_a.id)]
    assert str(chunk_b.id) not in turn.retrieved_chunk_ids
    assert chunk_b.text not in turn.answer
    cited_ids = {c["chunk_id"] for c in turn.citation_verifications}
    assert str(chunk_b.id) not in cited_ids

    # Path 3: citation verification given a citation that does name a
    # real chunk belonging to the other workspace: an explicit, named
    # refusal, not a result that merely happens to come back empty.
    cross_workspace_result = await verify_citations(
        db_session, workspace_id=workspace_a.id, cited_chunks=[chunk_b]
    )
    assert len(cross_workspace_result) == 1
    assert cross_workspace_result[0].verified is False
    assert cross_workspace_result[0].reason == "workspace_mismatch"

    # The reverse direction too, so none of this is an artifact of
    # insertion order or which workspace happened to be created first.
    results_b = await search_chunks(
        db_session,
        workspace_id=workspace_b.id,
        strategy=ChunkStrategy.NAIVE,
        query="q",
        embedder=embedder,
    )
    assert [r.id for r in results_b] == [chunk_b.id]
    cross_workspace_result_reverse = await verify_citations(
        db_session, workspace_id=workspace_b.id, cited_chunks=[chunk_a]
    )
    assert cross_workspace_result_reverse[0].verified is False
    assert cross_workspace_result_reverse[0].reason == "workspace_mismatch"


async def test_turn_workspace_id_always_matches_its_own_conversation_across_interleaved_calls(
    db_session: AsyncSession,
) -> None:
    """Turn.workspace_id is denormalized off Conversation.workspace_id
    (models.py's own module docstring explains why: every isolation
    sensitive query filters on Turn.workspace_id directly rather than
    joining through conversation first), which only stays safe if it can
    never drift from the conversation it was actually asked in. Interleaves
    turns across two conversations in two different workspaces and confirms
    every persisted Turn's workspace_id matches its own conversation's,
    never the other one's, regardless of call order.
    """
    embedder = _StubEmbedder(_vec(1.0, 0.0))
    workspace_a, _ = await _seed_workspace(
        db_session, name="interleave-a", chunk_text="a content", embedding=_vec(1.0, 0.0)
    )
    workspace_b, _ = await _seed_workspace(
        db_session, name="interleave-b", chunk_text="b content", embedding=_vec(1.0, 0.0)
    )
    conversation_a = Conversation(workspace_id=workspace_a.id)
    conversation_b = Conversation(workspace_id=workspace_b.id)
    db_session.add(conversation_a)
    db_session.add(conversation_b)
    await db_session.commit()

    turn_a1 = await ask(
        db_session,
        conversation_id=conversation_a.id,
        question="q1",
        strategy=ChunkStrategy.NAIVE,
        embedder=embedder,
        generator=ExtractiveGenerator(),
    )
    turn_b1 = await ask(
        db_session,
        conversation_id=conversation_b.id,
        question="q1",
        strategy=ChunkStrategy.NAIVE,
        embedder=embedder,
        generator=ExtractiveGenerator(),
    )
    turn_a2 = await ask(
        db_session,
        conversation_id=conversation_a.id,
        question="q2",
        strategy=ChunkStrategy.NAIVE,
        embedder=embedder,
        generator=ExtractiveGenerator(),
    )

    assert turn_a1.workspace_id == workspace_a.id
    assert turn_a2.workspace_id == workspace_a.id
    assert turn_b1.workspace_id == workspace_b.id
