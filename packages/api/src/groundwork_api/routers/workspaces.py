"""Workspaces: list the preloaded demo workspaces (zero setup, matches
section 16's "in one click, no setup, no key required"), upload a fresh
PDF into a brand new one, add another document to an existing workspace,
and (admin only) delete one.

Upload runs the real pipeline end to end, extract, both chunking
strategies, embed, index, the same path scripts/seed_demo_workspaces.py
uses for the preloaded demo content, never a shortcut taken because this
one call happens to be synchronous with an HTTP response. extract_pdf and
the table pass are CPU bound and synchronous; both run through
anyio.to_thread.run_sync so a large PDF cannot block the event loop for
every other in-flight request while it works.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from uuid import UUID

import anyio
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from groundwork_api.deps import get_session, require_admin_token
from groundwork_api.logging import get_logger
from groundwork_api.models import (
    Chunk,
    ChunkStrategy,
    Conversation,
    Document,
    EvalQuestion,
    EvalRun,
    ExtractionMethod,
    Turn,
    Workspace,
)
from groundwork_api.schemas import DocumentOut, WorkspaceDetailOut, WorkspaceOut
from groundwork_core.redaction import content_hash
from groundwork_ingest.extract import extract_pdf
from groundwork_retrieve.index import index_document

router = APIRouter(prefix="/workspaces", tags=["workspaces"])
logger = get_logger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB: generous for a demo PDF, not unbounded
PDF_MAGIC = b"%PDF-"


def _document_out(document: Document) -> DocumentOut:
    return DocumentOut(
        id=document.id,
        filename=document.filename,
        page_count=document.page_count,
        extraction_method=document.extraction_method,
        uploaded_at=document.uploaded_at,
    )


async def _workspace_detail(session: AsyncSession, workspace: Workspace) -> WorkspaceDetailOut:
    documents = (
        await session.execute(
            select(Document)
            .where(col(Document.workspace_id) == workspace.id)
            .order_by(col(Document.uploaded_at))
        )
    ).scalars()
    strategy_counts = await session.execute(
        select(col(Chunk.strategy), func.count())
        .where(col(Chunk.workspace_id) == workspace.id)
        .group_by(col(Chunk.strategy))
    )
    # dict(Row) trips mypy strict (Row isn't a plain tuple to its overloads
    # even though it behaves like one), so index each row explicitly.
    counts_by_strategy: dict[ChunkStrategy, int] = {row[0]: row[1] for row in strategy_counts.all()}
    return WorkspaceDetailOut(
        id=workspace.id,
        name=workspace.name,
        description=workspace.description,
        created_at=workspace.created_at,
        documents=[_document_out(d) for d in documents],
        naive_chunk_count=counts_by_strategy.get(ChunkStrategy.NAIVE, 0),
        structure_chunk_count=counts_by_strategy.get(ChunkStrategy.STRUCTURE, 0),
    )


async def _get_workspace_or_404(session: AsyncSession, workspace_id: UUID) -> Workspace:
    workspace = await session.get(Workspace, workspace_id)
    if workspace is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="workspace not found")
    return workspace


async def _ingest_upload(
    session: AsyncSession, *, workspace_id: UUID, upload: UploadFile
) -> Document:
    if upload.content_type not in (None, "application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"unsupported content type: {upload.content_type}",
        )

    body = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"file exceeds the {MAX_UPLOAD_BYTES} byte limit",
        )
    if not body.startswith(PDF_MAGIC):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="file does not look like a PDF"
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(body)
        tmp.flush()
        extracted = await anyio.to_thread.run_sync(extract_pdf, Path(tmp.name))

    document = Document(
        workspace_id=workspace_id,
        filename=upload.filename or "upload.pdf",
        sha256=extracted.sha256,
        page_count=extracted.page_count,
        extraction_method=ExtractionMethod(extracted.extraction_method),
    )
    session.add(document)
    await session.flush()

    await index_document(
        session, workspace_id=workspace_id, document_id=document.id, extracted=extracted
    )
    return document


@router.get("", response_model=list[WorkspaceOut])
async def list_workspaces(session: AsyncSession = Depends(get_session)) -> list[Workspace]:
    result = await session.execute(select(Workspace).order_by(col(Workspace.created_at)))
    return list(result.scalars())


@router.get("/{workspace_id}", response_model=WorkspaceDetailOut)
async def get_workspace(
    workspace_id: UUID, session: AsyncSession = Depends(get_session)
) -> WorkspaceDetailOut:
    workspace = await _get_workspace_or_404(session, workspace_id)
    return await _workspace_detail(session, workspace)


@router.post("", response_model=WorkspaceDetailOut, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    file: UploadFile = File(...),
    name: str | None = Form(default=None),
    session: AsyncSession = Depends(get_session),
) -> WorkspaceDetailOut:
    workspace = Workspace(name=name or file.filename or "Uploaded document")
    session.add(workspace)
    await session.flush()

    await _ingest_upload(session, workspace_id=workspace.id, upload=file)
    await session.commit()
    await session.refresh(workspace)
    return await _workspace_detail(session, workspace)


@router.post(
    "/{workspace_id}/documents",
    response_model=WorkspaceDetailOut,
    status_code=status.HTTP_201_CREATED,
)
async def add_document(
    workspace_id: UUID,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> WorkspaceDetailOut:
    workspace = await _get_workspace_or_404(session, workspace_id)
    await _ingest_upload(session, workspace_id=workspace.id, upload=file)
    await session.commit()
    return await _workspace_detail(session, workspace)


@router.delete(
    "/{workspace_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_admin_token)],
)
async def delete_workspace(
    workspace_id: UUID, session: AsyncSession = Depends(get_session)
) -> None:
    """Admin only (see deps.require_admin_token): erases a workspace and
    everything scoped to it. Deletes in dependency order, children first,
    each one scoped directly by the denormalized workspace_id column
    models.py's own module docstring describes, rather than joining
    through Conversation or Document to find rows to remove.
    """
    workspace = await _get_workspace_or_404(session, workspace_id)
    name_hash = content_hash(workspace.name)  # never log the real name, only its fingerprint

    await session.execute(delete(Turn).where(col(Turn.workspace_id) == workspace_id))
    await session.execute(
        delete(Conversation).where(col(Conversation.workspace_id) == workspace_id)
    )
    await session.execute(delete(Chunk).where(col(Chunk.workspace_id) == workspace_id))
    await session.execute(delete(Document).where(col(Document.workspace_id) == workspace_id))
    await session.execute(
        delete(EvalQuestion).where(col(EvalQuestion.workspace_id) == workspace_id)
    )
    await session.execute(delete(EvalRun).where(col(EvalRun.workspace_id) == workspace_id))
    await session.execute(delete(Workspace).where(col(Workspace.id) == workspace_id))
    await session.commit()
    logger.info("workspace.deleted", workspace_id=str(workspace_id), name_sha256_16=name_hash)
