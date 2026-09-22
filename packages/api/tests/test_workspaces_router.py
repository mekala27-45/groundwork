"""groundwork_api.routers.workspaces tests, over real HTTP: list, get,
upload (a real PDF through the real extract, chunk, embed, index
pipeline, not a hand typed byte string that merely satisfies the magic
byte check), add a second document, and the admin token gated delete,
its 503 with nothing configured, 401 with the wrong token, and 204 with
the right one, plus proof the cascade actually removed the workspace's
chunks and not merely the workspace row.

Deletion verification uses a fresh select rather than session.get: get()
is documented to answer from the identity map without a query when the
primary key is already loaded there, which is exactly true here since
this same db_session loaded these rows while seeding, so only a query
that genuinely reaches the database is trustworthy proof the row is gone.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from groundwork_api.models import Chunk, ChunkStrategy, Document, ExtractionMethod, Workspace
from groundwork_core.ids import new_id
from groundwork_ingest.fixtures import build_page_boundary_test_pdf

pytestmark = pytest.mark.requires_postgres


def _vec(first: float, second: float) -> list[float]:
    v = [0.0] * 384
    v[0] = first
    v[1] = second
    return v


async def _seed_workspace(session: AsyncSession, *, name: str = "api-test") -> Workspace:
    workspace = Workspace(name=name)
    session.add(workspace)
    await session.commit()
    return workspace


async def _seed_workspace_with_chunk(session: AsyncSession) -> tuple[Workspace, Chunk]:
    workspace = await _seed_workspace(session, name="api-chunk-test")
    document = Document(
        workspace_id=workspace.id,
        filename="doc.pdf",
        sha256=new_id().hex * 2,
        page_count=1,
        extraction_method=ExtractionMethod.TEXT,
    )
    session.add(document)
    await session.flush()
    text = "Sessions run fifty minutes."
    chunk = Chunk(
        document_id=document.id,
        workspace_id=workspace.id,
        strategy=ChunkStrategy.NAIVE,
        text=text,
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(text),
        embedding=_vec(1.0, 0.0),
        embedding_backend="tfidf",
    )
    session.add(chunk)
    await session.commit()
    return workspace, chunk


async def test_list_workspaces_is_empty_with_nothing_seeded(client: AsyncClient) -> None:
    response = await client.get("/workspaces")
    assert response.status_code == 200
    assert response.json() == []


async def test_list_workspaces_returns_a_seeded_workspace(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    workspace = await _seed_workspace(db_session)

    response = await client.get("/workspaces")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["id"] == str(workspace.id)
    assert body[0]["name"] == "api-test"


async def test_get_workspace_404s_when_it_does_not_exist(client: AsyncClient) -> None:
    response = await client.get(f"/workspaces/{new_id()}")
    assert response.status_code == 404


async def test_get_workspace_returns_documents_and_chunk_counts_by_strategy(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    workspace, _ = await _seed_workspace_with_chunk(db_session)

    response = await client.get(f"/workspaces/{workspace.id}")

    assert response.status_code == 200
    body = response.json()
    assert len(body["documents"]) == 1
    assert body["naive_chunk_count"] == 1
    assert body["structure_chunk_count"] == 0


async def test_create_workspace_uploads_a_real_pdf_and_indexes_it_end_to_end(
    client: AsyncClient, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "upload.pdf"
    build_page_boundary_test_pdf(pdf_path)

    response = await client.post(
        "/workspaces",
        files={"file": ("boundary.pdf", pdf_path.read_bytes(), "application/pdf")},
        data={"name": "Uploaded Workspace"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Uploaded Workspace"
    assert len(body["documents"]) == 1
    assert body["documents"][0]["page_count"] == 2
    # index_document always runs both strategies; chunk_structure falls
    # back to one whole document section when it finds no headings, so
    # both counts must be positive for this fixture's real body text.
    assert body["naive_chunk_count"] > 0
    assert body["structure_chunk_count"] > 0


async def test_create_workspace_defaults_its_name_to_the_filename_when_none_is_given(
    client: AsyncClient, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "upload.pdf"
    build_page_boundary_test_pdf(pdf_path)

    response = await client.post(
        "/workspaces",
        files={"file": ("unnamed.pdf", pdf_path.read_bytes(), "application/pdf")},
    )

    assert response.status_code == 201
    assert response.json()["name"] == "unnamed.pdf"


async def test_create_workspace_rejects_a_file_that_is_not_a_pdf(client: AsyncClient) -> None:
    response = await client.post(
        "/workspaces",
        files={"file": ("not-a-pdf.pdf", b"this is not a pdf", "application/pdf")},
    )
    assert response.status_code == 400


async def test_create_workspace_rejects_an_unsupported_content_type(client: AsyncClient) -> None:
    response = await client.post(
        "/workspaces",
        files={"file": ("image.png", b"\x89PNG\r\n\x1a\n" + b"0" * 20, "image/png")},
    )
    assert response.status_code == 400
    assert "content type" in response.json()["detail"]


async def test_create_workspace_rejects_a_file_over_the_size_limit(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("groundwork_api.routers.workspaces.MAX_UPLOAD_BYTES", 10)

    response = await client.post(
        "/workspaces",
        files={"file": ("big.pdf", b"%PDF-" + b"0" * 20, "application/pdf")},
    )

    assert response.status_code == 413


async def test_add_document_to_an_existing_workspace(
    client: AsyncClient, db_session: AsyncSession, tmp_path: Path
) -> None:
    workspace = await _seed_workspace(db_session)
    pdf_path = tmp_path / "second.pdf"
    build_page_boundary_test_pdf(pdf_path)

    response = await client.post(
        f"/workspaces/{workspace.id}/documents",
        files={"file": ("second.pdf", pdf_path.read_bytes(), "application/pdf")},
    )

    assert response.status_code == 201
    assert len(response.json()["documents"]) == 1


async def test_add_document_404s_when_the_workspace_does_not_exist(
    client: AsyncClient, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "orphan.pdf"
    build_page_boundary_test_pdf(pdf_path)

    response = await client.post(
        f"/workspaces/{new_id()}/documents",
        files={"file": ("orphan.pdf", pdf_path.read_bytes(), "application/pdf")},
    )

    assert response.status_code == 404


async def test_delete_workspace_without_any_admin_token_configured_is_503(
    client: AsyncClient, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GROUNDWORK_ADMIN_TOKEN", raising=False)
    workspace = await _seed_workspace(db_session)

    response = await client.delete(f"/workspaces/{workspace.id}")

    assert response.status_code == 503


async def test_delete_workspace_with_the_wrong_token_is_401(
    client: AsyncClient, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROUNDWORK_ADMIN_TOKEN", "correct-token")
    workspace = await _seed_workspace(db_session)

    response = await client.delete(
        f"/workspaces/{workspace.id}", headers={"X-Admin-Token": "wrong-token"}
    )

    assert response.status_code == 401


async def test_delete_workspace_with_the_correct_token_removes_it_and_its_chunks(
    client: AsyncClient, db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROUNDWORK_ADMIN_TOKEN", "correct-token")
    workspace, chunk = await _seed_workspace_with_chunk(db_session)

    response = await client.delete(
        f"/workspaces/{workspace.id}", headers={"X-Admin-Token": "correct-token"}
    )
    assert response.status_code == 204

    remaining_workspace = (
        await db_session.execute(select(Workspace).where(col(Workspace.id) == workspace.id))
    ).scalar_one_or_none()
    remaining_chunk = (
        await db_session.execute(select(Chunk).where(col(Chunk.id) == chunk.id))
    ).scalar_one_or_none()
    assert remaining_workspace is None
    assert remaining_chunk is None


async def test_delete_workspace_404s_when_it_does_not_exist(
    client: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROUNDWORK_ADMIN_TOKEN", "correct-token")

    response = await client.delete(
        f"/workspaces/{new_id()}", headers={"X-Admin-Token": "correct-token"}
    )

    assert response.status_code == 404
