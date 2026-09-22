"""Citation span verification tests, against a real pgvector backed
database: chunk existence, workspace scoping, and the sha256 comparison
section 10 asks for specifically ("a hash comparison, not a string
search"). test_fabricated_citation_detected is section 10's own named
test: "construct a turn record with a citation pointing at a chunk id
that does not exist, and assert the verification step flags it."
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from groundwork_api.models import Chunk, ChunkStrategy, Document, ExtractionMethod, Workspace
from groundwork_core.ids import new_id
from groundwork_verify.citation import CitationVerification, verify_citations

pytestmark = pytest.mark.requires_postgres


async def _seed_chunk(
    session: AsyncSession, *, workspace: Workspace, text: str = "some text"
) -> Chunk:
    document = Document(
        workspace_id=workspace.id,
        filename="doc.pdf",
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
        text=text,
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(text),
    )
    session.add(chunk)
    await session.commit()
    return chunk


async def test_citation_is_verified_when_the_chunk_is_unchanged(db_session: AsyncSession) -> None:
    workspace = Workspace(name="citation-test")
    db_session.add(workspace)
    await db_session.flush()
    chunk = await _seed_chunk(db_session, workspace=workspace)

    results = await verify_citations(db_session, workspace_id=workspace.id, cited_chunks=[chunk])

    assert results == [CitationVerification(chunk_id=chunk.id, verified=True, reason=None)]


async def test_fabricated_citation_detected(db_session: AsyncSession) -> None:
    """Section 10's own named test: a citation pointing at a chunk id
    that was never persisted at all, the sharpest version of a fabricated
    citation, must be flagged rather than pass silently.
    """
    workspace = Workspace(name="citation-test")
    db_session.add(workspace)
    await db_session.flush()

    fabricated = Chunk(
        id=new_id(),
        document_id=new_id(),
        workspace_id=workspace.id,
        strategy=ChunkStrategy.NAIVE,
        text="this chunk was never actually saved to the database",
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=10,
    )

    results = await verify_citations(
        db_session, workspace_id=workspace.id, cited_chunks=[fabricated]
    )

    assert len(results) == 1
    assert results[0].verified is False
    assert results[0].reason == "chunk_not_found"


async def test_citation_flagged_when_the_chunk_belongs_to_another_workspace(
    db_session: AsyncSession,
) -> None:
    real_workspace = Workspace(name="real-workspace")
    other_workspace = Workspace(name="other-workspace")
    db_session.add(real_workspace)
    db_session.add(other_workspace)
    await db_session.flush()
    chunk = await _seed_chunk(db_session, workspace=real_workspace)

    # A citation claiming to belong to a conversation in other_workspace,
    # naming a chunk that actually lives in real_workspace: exactly the
    # cross tenant leak test_workspace_isolation checks for elsewhere in
    # this codebase, at the citation layer specifically.
    results = await verify_citations(
        db_session, workspace_id=other_workspace.id, cited_chunks=[chunk]
    )

    assert results[0].verified is False
    assert results[0].reason == "workspace_mismatch"


async def test_citation_flagged_when_the_chunk_text_changed_since_generation(
    db_session: AsyncSession,
) -> None:
    """The hash comparison section 10 specifically asks for: a chunk
    edited or re-indexed after the fact must be caught even though its id
    still exists and still belongs to the right workspace.
    """
    workspace = Workspace(name="citation-test")
    db_session.add(workspace)
    await db_session.flush()
    original_text = "the original text at generation time"
    chunk = await _seed_chunk(db_session, workspace=workspace, text=original_text)

    # cited_at_generation_time is a snapshot of the chunk exactly as it
    # was passed to the generator: same id, same workspace, but the text
    # a later edit is about to change, built as a fresh, detached object
    # rather than by mutating and copying the tracked one.
    cited_at_generation_time = Chunk(
        id=chunk.id,
        document_id=chunk.document_id,
        workspace_id=chunk.workspace_id,
        strategy=chunk.strategy,
        text=original_text,
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(original_text),
    )
    chunk.text = "an edited replacement that changes the stored content"
    db_session.add(chunk)
    await db_session.commit()

    results = await verify_citations(
        db_session, workspace_id=workspace.id, cited_chunks=[cited_at_generation_time]
    )

    assert results[0].verified is False
    assert results[0].reason == "text_hash_mismatch"


async def test_verify_citations_scores_several_chunks_independently(
    db_session: AsyncSession,
) -> None:
    workspace = Workspace(name="citation-test")
    db_session.add(workspace)
    await db_session.flush()
    good = await _seed_chunk(
        db_session, workspace=workspace, text="a citation that will verify fine"
    )
    fabricated = Chunk(
        id=new_id(),
        document_id=new_id(),
        workspace_id=workspace.id,
        strategy=ChunkStrategy.NAIVE,
        text="never persisted",
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=10,
    )

    results = await verify_citations(
        db_session, workspace_id=workspace.id, cited_chunks=[good, fabricated]
    )

    assert [r.verified for r in results] == [True, False]
    assert [r.chunk_id for r in results] == [good.id, fabricated.id]
