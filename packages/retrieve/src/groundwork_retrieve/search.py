"""Vector search against pgvector: workspace and strategy scoped, as
section 8 of the day 5 prompt specifies, so the retrieval evaluation
harness (still to come) can score naive against structure aware chunking
under identical conditions.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from groundwork_api.models import Chunk
from groundwork_api.models import ChunkStrategy as DbChunkStrategy
from groundwork_retrieve.embeddings import Embedder, get_embedder


async def search_chunks(
    session: AsyncSession,
    *,
    workspace_id: UUID,
    strategy: DbChunkStrategy,
    query: str,
    top_k: int = 8,
    embedder: Embedder | None = None,
) -> list[Chunk]:
    """Embeds query with the same embedder chunks were indexed with (the
    caller is responsible for that consistency: mixing backends between
    index time and query time is a modeling error the type system cannot
    catch, only config discipline can) and returns the top_k nearest chunks
    in this workspace and strategy by cosine distance.
    """
    embedder = embedder or get_embedder()
    [query_vector] = embedder.embed([query])

    distance = Chunk.embedding.cosine_distance(query_vector)  # type: ignore[union-attr]
    result = await session.execute(
        select(Chunk)
        .where(col(Chunk.workspace_id) == workspace_id)
        .where(col(Chunk.strategy) == strategy)
        .where(col(Chunk.embedding).is_not(None))
        .order_by(distance)
        .limit(top_k)
    )
    return list(result.scalars().all())
