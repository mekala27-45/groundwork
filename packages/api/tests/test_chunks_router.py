"""groundwork_api.routers.chunks tests: the by-id chunk lookup /chat and
/trace (build order steps 22 and 23) both depend on to render a citation
as its actual passage and page number rather than a bare id.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from groundwork_api.models import (
    Chunk,
    ChunkStrategy,
    Document,
    ExtractionMethod,
    Workspace,
)
from groundwork_core.ids import new_id

pytestmark = pytest.mark.requires_postgres


async def _seed_chunk(session: AsyncSession) -> Chunk:
    workspace = Workspace(name="chunk-router-test")
    session.add(workspace)
    await session.flush()
    document = Document(
        workspace_id=workspace.id,
        filename="doc.pdf",
        sha256=new_id().hex * 2,
        page_count=2,
        extraction_method=ExtractionMethod.TEXT,
    )
    session.add(document)
    await session.flush()
    chunk = Chunk(
        document_id=document.id,
        workspace_id=workspace.id,
        strategy=ChunkStrategy.STRUCTURE,
        text="Sessions run fifty minutes and take place over video by default.",
        page_start=1,
        page_end=2,
        char_start=0,
        char_end=66,
        section_title="How Sessions Work",
    )
    session.add(chunk)
    await session.commit()
    return chunk


async def test_get_chunk_returns_the_seeded_chunk(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    chunk = await _seed_chunk(db_session)

    response = await client.get(f"/chunks/{chunk.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(chunk.id)
    assert body["document_id"] == str(chunk.document_id)
    assert body["workspace_id"] == str(chunk.workspace_id)
    assert body["strategy"] == "structure"
    assert body["text"] == chunk.text
    assert body["page_start"] == 1
    assert body["page_end"] == 2
    assert body["section_title"] == "How Sessions Work"
    assert body["injection_flag"] is None


async def test_get_chunk_404s_on_an_id_nothing_ever_stored(client: AsyncClient) -> None:
    response = await client.get(f"/chunks/{new_id()}")

    assert response.status_code == 404
    assert response.json()["detail"] == "chunk not found"
