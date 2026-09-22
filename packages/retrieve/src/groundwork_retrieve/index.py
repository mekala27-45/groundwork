"""Turns chunk candidates into stored, embedded Chunk rows: the pgvector
indexing step build order step 8 asks for, run once per document across
both chunking strategies so the retrieval evaluation harness (still to
come) can score naive against structure aware chunking on exactly the
same underlying document rather than two separately prepared ones.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from groundwork_api.models import Chunk
from groundwork_api.models import ChunkStrategy as DbChunkStrategy
from groundwork_chunk import ChunkCandidate, chunk_naive, chunk_structure
from groundwork_ingest.models import ExtractedDocument
from groundwork_retrieve.embeddings import Embedder, get_embedder, resolve_embedding_backend


async def index_document(
    session: AsyncSession,
    *,
    workspace_id: UUID,
    document_id: UUID,
    extracted: ExtractedDocument,
    embedder: Embedder | None = None,
) -> list[Chunk]:
    """Runs both chunking strategies against extracted, embeds every
    resulting chunk, and adds one Chunk row per candidate to session:
    flushed so the rows have ids and are visible within the current
    transaction, but not committed, the same division of responsibility
    every other write in this codebase follows (the caller controls the
    transaction boundary). Returns the rows so a caller can report how
    many landed per strategy without a second query.

    embedder defaults to the process wide get_embedder(); tests pass a
    small stub instead, both to stay fast and to keep this function's own
    tests independent of whichever backend this environment's network
    happens to resolve "auto" to.
    """
    embedder = embedder or get_embedder()
    backend = resolve_embedding_backend()

    rows: list[Chunk] = []
    for strategy, candidates in (
        (DbChunkStrategy.NAIVE, chunk_naive(extracted)),
        (DbChunkStrategy.STRUCTURE, chunk_structure(extracted)),
    ):
        rows.extend(
            _embed_and_build(
                candidates,
                embedder=embedder,
                backend=backend,
                strategy=strategy,
                workspace_id=workspace_id,
                document_id=document_id,
            )
        )

    for row in rows:
        session.add(row)
    await session.flush()
    return rows


def _embed_and_build(
    candidates: list[ChunkCandidate],
    *,
    embedder: Embedder,
    backend: str,
    strategy: DbChunkStrategy,
    workspace_id: UUID,
    document_id: UUID,
) -> list[Chunk]:
    if not candidates:
        return []
    vectors = embedder.embed([candidate.text for candidate in candidates])
    return [
        Chunk(
            document_id=document_id,
            workspace_id=workspace_id,
            strategy=strategy,
            text=candidate.text,
            page_start=candidate.page_start,
            page_end=candidate.page_end,
            char_start=candidate.char_start,
            char_end=candidate.char_end,
            section_title=candidate.section_title,
            embedding=vector,
            embedding_backend=backend,
        )
        for candidate, vector in zip(candidates, vectors, strict=True)
    ]
