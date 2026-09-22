"""Seeds the demo workspaces this project ships preloaded, no upload
required: Meridian Coaching, the ML evaluation reference guide, and a
third, deliberately separate workspace holding only the injection red
team fixture.

Every document is extracted with the real ingestion pipeline
(groundwork_ingest.extract.extract_pdf), then indexed with both chunking
strategies through groundwork_retrieve.index.index_document, exactly the
path a real upload takes. Nothing here is a shortcut or a pre-baked
fixture: the Chunk rows this produces, and their ids, are what
evalset/questions.yaml's expected_chunk_ids actually references, and what
scripts/run_eval.py (build order step 11) will score retrieval against.

The injection test fixture gets its own workspace rather than joining
Meridian's, deliberately. build_injection_test_pdf's visible content (a
generic business FAQ) is intentionally similar in shape to Meridian's own
real FAQ document, since a convincing attack looks like an ordinary
document, not an obviously fake one. Indexed into the same workspace as
the real FAQ, the two would compete for the same queries and make
retrieval of the planted chunk non-deterministic, which is exactly the
wrong property for a test whose entire point is a hard, repeatable
pass/fail. Alone in its own workspace, whatever question is asked has
only that document's chunks to retrieve from.

Run with: uv run python scripts/seed_demo_workspaces.py
Safe to re-run: it truncates its own three workspaces (by name) first, so
re-running after a chunking or content change never leaves stale rows
behind next to fresh ones.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from groundwork_api.db import dispose_engine, session_scope
from groundwork_api.models import (
    Chunk,
    Conversation,
    Document,
    EvalQuestion,
    EvalRun,
    ExtractionMethod,
    Turn,
    Workspace,
)
from groundwork_ingest.extract import extract_pdf
from groundwork_ingest.fixtures import INJECTION_TEST_MARKER
from groundwork_retrieve.index import index_document

REPO_ROOT = Path(__file__).resolve().parent.parent

MERIDIAN_NAME = "Meridian Coaching"
REFERENCE_GUIDE_NAME = "ML Evaluation Reference Guide"
INJECTION_WORKSPACE_NAME = "Injection Red Team Fixture"

MERIDIAN_FILES = ["methodology.pdf", "packages.pdf", "faq.pdf", "onboarding.pdf"]


async def _reset_workspace(session: AsyncSession, name: str, description: str) -> Workspace:
    """Deletes any workspace with this name and everything scoped to it,
    then inserts a fresh row. Every table models.py's own module docstring
    lists as workspace scoped (Turn and Conversation included, both
    denormalized the same way Chunk is) is cleaned up here, in the exact
    dependency order, children first, that
    groundwork_api.routers.workspaces.delete_workspace already uses and
    already has a real end to end test for: this function is that
    same cleanup, not an independently maintained copy of it that could
    quietly drift out of sync.

    A real gap this session found the hard way, not a hypothetical:
    run_eval.py (build order step 20) was the first thing in this build
    to ever create real Conversation and Turn rows against these three
    demo workspaces, and the very next reseed after that failed outright
    with a foreign key violation, "conversation_workspace_id_fkey", since
    this function had never deleted either table. Fixed here rather than
    only in the one run that happened to hit it, since every future
    reseed after any real chat activity would fail the identical way.
    """
    existing = (
        (await session.execute(select(Workspace).where(Workspace.name == name))).scalars().all()
    )
    for old in existing:
        await session.execute(delete(Turn).where(Turn.workspace_id == old.id))
        await session.execute(delete(Conversation).where(Conversation.workspace_id == old.id))
        await session.execute(delete(Chunk).where(Chunk.workspace_id == old.id))
        await session.execute(delete(Document).where(Document.workspace_id == old.id))
        await session.execute(delete(EvalQuestion).where(EvalQuestion.workspace_id == old.id))
        await session.execute(delete(EvalRun).where(EvalRun.workspace_id == old.id))
        await session.delete(old)
    await session.flush()

    workspace = Workspace(name=name, description=description)
    session.add(workspace)
    await session.flush()
    return workspace


async def _ingest_and_index(
    session: AsyncSession, *, workspace: Workspace, pdf_path: Path
) -> tuple[Document, list[Chunk]]:
    extracted = extract_pdf(pdf_path)
    document = Document(
        workspace_id=workspace.id,
        filename=pdf_path.name,
        sha256=extracted.sha256,
        page_count=extracted.page_count,
        extraction_method=ExtractionMethod(extracted.extraction_method),
    )
    session.add(document)
    await session.flush()

    chunks = await index_document(
        session,
        workspace_id=workspace.id,
        document_id=document.id,
        extracted=extracted,
    )
    return document, chunks


async def main() -> None:
    async with session_scope() as session:
        meridian = await _reset_workspace(
            session,
            MERIDIAN_NAME,
            "A fictional life and business coaching practice, invented for this project. "
            "Every figure in its four source documents is made up but internally consistent.",
        )
        reference_guide = await _reset_workspace(
            session,
            REFERENCE_GUIDE_NAME,
            "An original technical reference on evaluating machine learning systems, written "
            "for this project. See evalset/public_domain/PROVENANCE.md for why this workspace "
            "does not hold a downloaded government publication as originally specified.",
        )
        injection_workspace = await _reset_workspace(
            session,
            INJECTION_WORKSPACE_NAME,
            "Holds only the indirect prompt injection red team fixture, isolated in its own "
            "workspace so retrieval against it is deterministic. Not part of the two preloaded "
            "chat demo workspaces; used by the injection defense test and the eval harness's "
            "injection category only.",
        )

        summary: list[str] = []

        for filename in MERIDIAN_FILES:
            document, chunks = await _ingest_and_index(
                session, workspace=meridian, pdf_path=REPO_ROOT / "evalset" / "meridian" / filename
            )
            summary.append(
                f"{MERIDIAN_NAME} / {filename}: document={document.id} chunks={len(chunks)}"
            )

        document, chunks = await _ingest_and_index(
            session,
            workspace=reference_guide,
            pdf_path=REPO_ROOT / "evalset" / "public_domain" / "ml-eval-reference-guide.pdf",
        )
        summary.append(
            f"{REFERENCE_GUIDE_NAME} / ml-eval-reference-guide.pdf: document={document.id} "
            f"chunks={len(chunks)}"
        )

        document, chunks = await _ingest_and_index(
            session,
            workspace=injection_workspace,
            pdf_path=REPO_ROOT / "evalset" / "injection_test.pdf",
        )
        summary.append(
            f"{INJECTION_WORKSPACE_NAME} / injection_test.pdf: document={document.id} "
            f"chunks={len(chunks)}"
        )

        await session.commit()

        print(f"marker: {INJECTION_TEST_MARKER}")
        print(f"workspace: {MERIDIAN_NAME} = {meridian.id}")
        print(f"workspace: {REFERENCE_GUIDE_NAME} = {reference_guide.id}")
        print(f"workspace: {INJECTION_WORKSPACE_NAME} = {injection_workspace.id}")
        for line in summary:
            print(line)

    await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
