"""search_chunks tests: workspace and strategy scoped nearest neighbour
lookup against a real pgvector index. test_models.py's own
test_pgvector_cosine_distance_orders_by_similarity already proves the bare
<=> query orders correctly; these tests go through the retrieve package's
public function instead and add the workspace and strategy scoping that
bare query never had to prove, since a search that ignored either would
leak one tenant's or one chunking strategy's chunks into another's
results.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from groundwork_api.models import Chunk, ChunkStrategy, Document, ExtractionMethod, Workspace
from groundwork_retrieve.search import search_chunks

pytestmark = pytest.mark.requires_postgres


class _StubEmbedder:
    """Returns one fixed, caller supplied vector for any query text, so a
    test can pin down exactly which stored chunk should be nearest without
    depending on any real embedding model."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector for _ in texts]


def _vec(first: float, second: float) -> list[float]:
    v = [0.0] * 384
    v[0] = first
    v[1] = second
    return v


async def _seed(session: AsyncSession) -> Workspace:
    ws = Workspace(name="search-test")
    session.add(ws)
    await session.flush()
    doc = Document(
        workspace_id=ws.id,
        filename="d.pdf",
        sha256="2" * 64,
        page_count=1,
        extraction_method=ExtractionMethod.TEXT,
    )
    session.add(doc)
    await session.flush()

    naive_chunks = [
        Chunk(
            document_id=doc.id,
            workspace_id=ws.id,
            strategy=ChunkStrategy.NAIVE,
            text=text,
            page_start=1,
            page_end=1,
            char_start=0,
            char_end=len(text),
            embedding=vector,
            embedding_backend="tfidf",
        )
        for text, vector in [
            ("near", _vec(0.9, 1.0)),
            ("mid", _vec(0.3, 1.0)),
            ("far", _vec(-0.9, 1.0)),
        ]
    ]
    structure_chunk = Chunk(
        document_id=doc.id,
        workspace_id=ws.id,
        strategy=ChunkStrategy.STRUCTURE,
        text="wrong strategy",
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=1,
        embedding=_vec(1.0, 1.0),
        embedding_backend="tfidf",
    )
    for chunk in [*naive_chunks, structure_chunk]:
        session.add(chunk)
    await session.commit()
    return ws


async def test_search_returns_nearest_chunks_in_similarity_order(
    db_session: AsyncSession,
) -> None:
    ws = await _seed(db_session)

    results = await search_chunks(
        db_session,
        workspace_id=ws.id,
        strategy=ChunkStrategy.NAIVE,
        query="anything, the stub embedder ignores this",
        embedder=_StubEmbedder(_vec(1.0, 1.0)),
    )

    assert [r.text for r in results] == ["near", "mid", "far"]


async def test_search_is_scoped_to_the_requested_strategy(db_session: AsyncSession) -> None:
    ws = await _seed(db_session)

    results = await search_chunks(
        db_session,
        workspace_id=ws.id,
        strategy=ChunkStrategy.STRUCTURE,
        query="q",
        embedder=_StubEmbedder(_vec(1.0, 1.0)),
    )

    assert [r.text for r in results] == ["wrong strategy"]


async def test_search_is_scoped_to_the_requested_workspace(db_session: AsyncSession) -> None:
    await _seed(db_session)
    other = Workspace(name="other-workspace")
    db_session.add(other)
    await db_session.commit()

    results = await search_chunks(
        db_session,
        workspace_id=other.id,
        strategy=ChunkStrategy.NAIVE,
        query="q",
        embedder=_StubEmbedder(_vec(1.0, 1.0)),
    )

    assert results == []


async def test_search_respects_top_k(db_session: AsyncSession) -> None:
    ws = await _seed(db_session)

    results = await search_chunks(
        db_session,
        workspace_id=ws.id,
        strategy=ChunkStrategy.NAIVE,
        query="q",
        top_k=1,
        embedder=_StubEmbedder(_vec(1.0, 1.0)),
    )

    assert [r.text for r in results] == ["near"]


async def test_search_reaches_the_embedder_with_the_query_text(db_session: AsyncSession) -> None:
    """Did it run at all: the query text must actually reach the embedder,
    independent of whether the ranking it produces is correct (covered
    separately above)."""
    ws = await _seed(db_session)
    calls: list[str] = []

    class _RecordingEmbedder:
        def embed(self, texts: list[str]) -> list[list[float]]:
            calls.extend(texts)
            return [_vec(0.9, 1.0) for _ in texts]

    await search_chunks(
        db_session,
        workspace_id=ws.id,
        strategy=ChunkStrategy.NAIVE,
        query="a distinctive query string",
        embedder=_RecordingEmbedder(),
    )

    assert calls == ["a distinctive query string"]
