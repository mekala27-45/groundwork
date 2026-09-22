"""index_document tests: the orchestration function that turns one
ExtractedDocument into stored, embedded Chunk rows under both chunking
strategies. A tiny stub Embedder stands in for the real one so these tests
are fast, deterministic, and independent of whichever backend this
environment's network happens to resolve "auto" to, per index.py's own
docstring. As with the ocr and tokenizer resilience tests, "did it run"
(both strategies contributed, the embedder was actually called) and "did
it produce the right output" (correct scoping, real vectors, correct
backend label) are asserted separately rather than folded into one test.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from groundwork_api.models import Chunk, ChunkStrategy, Document, ExtractionMethod, Workspace
from groundwork_ingest.models import ExtractedDocument, ExtractedPage, ExtractedSpan
from groundwork_retrieve.embeddings import reset_embedder_cache
from groundwork_retrieve.index import index_document

pytestmark = pytest.mark.requires_postgres


@pytest.fixture(autouse=True)
def _reset_embedder_cache() -> Iterator[None]:
    reset_embedder_cache()
    yield
    reset_embedder_cache()


class _StubEmbedder:
    """Maps each text to a vector derived from its own length, so tests can
    assert on exact stored vectors without depending on a real model or the
    tfidf fallback's own hashing. Records every batch it was called with so
    tests can assert the orchestration actually reached the embedder."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text))] * 384 for text in texts]


def _span(text: str, *, page: int = 1) -> ExtractedSpan:
    return ExtractedSpan(
        text=text,
        page_number=page,
        bbox=(0.0, 0.0, 10.0, 10.0),
        font_size=11.0,
        color_rgb=(0, 0, 0),
        font_name="Helvetica",
        is_bold=False,
    )


def _document() -> ExtractedDocument:
    words = ["Meridian's", "onboarding", "begins", "with", "a", "discovery", "call", "today"]
    text = " ".join(words)
    page = ExtractedPage(
        page_number=1,
        text=text,
        spans=[_span(w) for w in words],
        tables=[],
        char_count=len(text),
        page_area_pt2=1000.0,
        text_density=0.05,
        used_ocr=False,
    )
    return ExtractedDocument(filename="onboarding.pdf", sha256="1" * 64, page_count=1, pages=[page])


async def _seed_workspace_and_document(session: AsyncSession) -> tuple[Workspace, Document]:
    ws = Workspace(name="index-test")
    session.add(ws)
    await session.flush()
    doc = Document(
        workspace_id=ws.id,
        filename="onboarding.pdf",
        sha256="1" * 64,
        page_count=1,
        extraction_method=ExtractionMethod.TEXT,
    )
    session.add(doc)
    await session.flush()
    return ws, doc


async def test_index_document_runs_both_strategies(db_session: AsyncSession) -> None:
    """Did the step run at all: both strategies must contribute rows and
    the embedder must be reached once per strategy, independent of whether
    the stored content turns out to be correct."""
    ws, doc = await _seed_workspace_and_document(db_session)
    embedder = _StubEmbedder()

    rows = await index_document(
        db_session,
        workspace_id=ws.id,
        document_id=doc.id,
        extracted=_document(),
        embedder=embedder,
    )

    strategies_seen = {row.strategy for row in rows}
    assert strategies_seen == {ChunkStrategy.NAIVE, ChunkStrategy.STRUCTURE}
    assert len(embedder.calls) == 2


async def test_index_document_rows_are_scoped_and_carry_real_vectors(
    db_session: AsyncSession,
) -> None:
    """Did it produce the right output: every row belongs to the right
    workspace and document, carries the vector the embedder actually
    returned for its own text, and records a valid backend label."""
    ws, doc = await _seed_workspace_and_document(db_session)
    embedder = _StubEmbedder()

    rows = await index_document(
        db_session,
        workspace_id=ws.id,
        document_id=doc.id,
        extracted=_document(),
        embedder=embedder,
    )

    assert rows
    for row in rows:
        assert row.workspace_id == ws.id
        assert row.document_id == doc.id
        assert row.embedding is not None
        assert len(row.embedding) == 384
        assert row.embedding_backend in ("local_model", "tfidf")
        assert row.embedding == [float(len(row.text))] * 384


async def test_index_document_flushes_without_committing(db_session: AsyncSession) -> None:
    """flush(), not commit(): rows must already be visible to a query
    inside the same transaction, without the caller committing first."""
    ws, doc = await _seed_workspace_and_document(db_session)

    rows = await index_document(
        db_session,
        workspace_id=ws.id,
        document_id=doc.id,
        extracted=_document(),
        embedder=_StubEmbedder(),
    )

    fetched = (
        (await db_session.execute(select(Chunk).where(col(Chunk.workspace_id) == ws.id)))
        .scalars()
        .all()
    )
    assert {c.id for c in fetched} == {row.id for row in rows}


async def test_index_document_uses_the_resolved_backend_when_no_embedder_is_injected(
    monkeypatch: pytest.MonkeyPatch, db_session: AsyncSession
) -> None:
    """No embedder argument falls back to the process wide get_embedder();
    forcing the backend to tfidf keeps this deterministic without
    depending on whether this environment can reach huggingface.co."""
    monkeypatch.setenv("GROUNDWORK_EMBEDDING_BACKEND", "tfidf")
    ws, doc = await _seed_workspace_and_document(db_session)

    rows = await index_document(
        db_session, workspace_id=ws.id, document_id=doc.id, extracted=_document()
    )

    assert rows
    assert all(row.embedding_backend == "tfidf" for row in rows)


async def test_index_document_on_an_empty_document_produces_no_rows(
    db_session: AsyncSession,
) -> None:
    ws, doc = await _seed_workspace_and_document(db_session)
    empty_page = ExtractedPage(
        page_number=1,
        text="",
        spans=[],
        tables=[],
        char_count=0,
        page_area_pt2=1000.0,
        text_density=0.0,
        used_ocr=False,
    )
    empty = ExtractedDocument(
        filename="empty.pdf", sha256="2" * 64, page_count=1, pages=[empty_page]
    )
    embedder = _StubEmbedder()

    rows = await index_document(
        db_session, workspace_id=ws.id, document_id=doc.id, extracted=empty, embedder=embedder
    )

    assert rows == []
    assert embedder.calls == []
