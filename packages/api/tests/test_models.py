from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from groundwork_api.models import (
    Chunk,
    ChunkStrategy,
    Document,
    EvalCategory,
    EvalQuestion,
    ExtractionMethod,
    Workspace,
)
from groundwork_core.ids import new_id

pytestmark = pytest.mark.requires_postgres


async def test_workspace_document_chunk_round_trip(db_session: AsyncSession) -> None:
    ws = Workspace(name="Meridian Coaching", description="fictional demo workspace")
    db_session.add(ws)
    await db_session.flush()

    doc = Document(
        workspace_id=ws.id,
        filename="methodology.pdf",
        sha256="a" * 64,
        page_count=12,
        extraction_method=ExtractionMethod.TEXT,
    )
    db_session.add(doc)
    await db_session.flush()

    chunk = Chunk(
        document_id=doc.id,
        workspace_id=ws.id,
        strategy=ChunkStrategy.NAIVE,
        text="Meridian's five stage onboarding begins with a discovery call.",
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=64,
        embedding=[0.01] * 384,
        embedding_backend="tfidf",
    )
    db_session.add(chunk)
    await db_session.commit()

    fetched = (
        await db_session.execute(select(Chunk).where(col(Chunk.workspace_id) == ws.id))
    ).scalar_one()
    assert fetched.text.startswith("Meridian's five stage onboarding")
    assert fetched.embedding is not None
    assert len(fetched.embedding) == 384
    assert fetched.embedding_backend == "tfidf"


async def test_pgvector_cosine_distance_orders_by_similarity(db_session: AsyncSession) -> None:
    """A real pgvector <=> query, not a mock: three chunks with hand-built
    embeddings, and the nearest-neighbour order is asserted against the
    vector actually stored, not against the text."""
    ws = Workspace(name="probe")
    db_session.add(ws)
    await db_session.flush()
    doc = Document(
        workspace_id=ws.id,
        filename="d.pdf",
        sha256="b" * 64,
        page_count=1,
        extraction_method=ExtractionMethod.TEXT,
    )
    db_session.add(doc)
    await db_session.flush()

    def vec(first: float) -> list[float]:
        v = [0.0] * 384
        v[0] = first
        v[1] = 1.0
        return v

    near = Chunk(
        document_id=doc.id,
        workspace_id=ws.id,
        strategy=ChunkStrategy.NAIVE,
        text="near",
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=1,
        embedding=vec(0.9),
    )
    mid = Chunk(
        document_id=doc.id,
        workspace_id=ws.id,
        strategy=ChunkStrategy.NAIVE,
        text="mid",
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=1,
        embedding=vec(0.3),
    )
    far = Chunk(
        document_id=doc.id,
        workspace_id=ws.id,
        strategy=ChunkStrategy.NAIVE,
        text="far",
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=1,
        embedding=vec(-0.9),
    )
    db_session.add_all([mid, far, near])  # inserted out of similarity order on purpose
    await db_session.commit()

    query_vec = vec(1.0)
    # pgvector's Comparator methods (cosine_distance and friends) live on the
    # runtime SQL expression that Chunk.embedding produces at class access
    # time; pgvector-sqlalchemy does not ship stubs that expose them through
    # SQLModel's Mapped[...] wrapper, so this one call needs the ignore.
    distance = Chunk.embedding.cosine_distance(query_vec)  # type: ignore[union-attr]
    rows = (
        (
            await db_session.execute(
                select(col(Chunk.text))
                .where(col(Chunk.workspace_id) == ws.id)
                .order_by(distance)
                .limit(3)
            )
        )
        .scalars()
        .all()
    )

    assert list(rows) == ["near", "mid", "far"]


async def test_eval_question_stores_expected_chunk_ids(db_session: AsyncSession) -> None:
    ws = Workspace(name="probe2")
    db_session.add(ws)
    await db_session.flush()
    expected = [str(new_id()), str(new_id())]
    q = EvalQuestion(
        workspace_id=ws.id,
        question="What does the onboarding process involve?",
        expected_chunk_ids=expected,
        category=EvalCategory.DIRECT,
    )
    db_session.add(q)
    await db_session.commit()

    fetched = (
        await db_session.execute(
            select(EvalQuestion).where(col(EvalQuestion.workspace_id) == ws.id)
        )
    ).scalar_one()
    assert fetched.expected_chunk_ids == expected
    assert fetched.category == EvalCategory.DIRECT
