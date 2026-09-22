"""A single chunk lookup by id: the exact passage a citation points at, the
gap building the web app to section 12's own spec surfaced. Section 12
requires /chat to expand a citation "to the exact highlighted source
passage on click, with its page number" and /trace to show "the retrieved
chunks", and Turn only ever stores retrieved_chunk_ids (bare id strings,
see chat.py's own module docstring for why), never the chunk text or page
number itself. Without this endpoint neither view could render anything
but an opaque UUID.

No workspace scoping on this lookup, deliberately, the same shape as
GET /turns/{turn_id} in conversations.py: both are by-id detail fetches
behind an unguessable UUIDv7, not a listing that could enumerate another
workspace's content. A citation link already carries the chunk id, so
requiring a matching workspace_id here would only add a parameter a
correct caller already implies and a hostile one could still supply,
without closing any path test_workspace_isolation covers, that test scopes
retrieval and chat, which this lookup is neither.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from groundwork_api.deps import get_session
from groundwork_api.models import Chunk
from groundwork_api.schemas import ChunkOut

router = APIRouter(tags=["chunks"])


@router.get("/chunks/{chunk_id}", response_model=ChunkOut)
async def get_chunk(chunk_id: UUID, session: AsyncSession = Depends(get_session)) -> ChunkOut:
    chunk = await session.get(Chunk, chunk_id)
    if chunk is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="chunk not found")
    return ChunkOut(
        id=chunk.id,
        document_id=chunk.document_id,
        workspace_id=chunk.workspace_id,
        strategy=chunk.strategy,
        text=chunk.text,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section_title=chunk.section_title,
        injection_flag=chunk.injection_flag,
    )
