from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from groundwork_api.claims import NOT_YET_RUN, build_manifest, render_template
from groundwork_api.models import (
    Chunk,
    ChunkStrategy,
    Document,
    EvalRun,
    ExtractionMethod,
    Workspace,
)
from groundwork_core.ids import new_id

pytestmark = pytest.mark.requires_postgres


def test_render_template_substitutes_every_known_placeholder() -> None:
    rendered = render_template(
        "Workspaces: {{workspace_count}}. Status: {{status}}.",
        {"workspace_count": 3, "status": "ok"},
    )
    assert rendered == "Workspaces: 3. Status: ok."


def test_render_template_raises_on_a_placeholder_the_manifest_does_not_cover() -> None:
    """The deliberate violation case: a template drifts ahead of the
    manifest (a typo, or a section added before its data exists) and the
    render fails loudly instead of shipping the literal {{placeholder}}
    text into a published document.
    """
    with pytest.raises(KeyError, match="typo_not_in_manifest"):
        render_template("{{typo_not_in_manifest}}", {"workspace_count": 1})


def test_render_template_on_empty_manifest_and_empty_template_is_a_no_op() -> None:
    assert render_template("", {}) == ""
    assert render_template("no placeholders here", {}) == "no placeholders here"


async def test_build_manifest_reports_not_yet_run_before_any_eval_run(
    db_session: AsyncSession,
) -> None:
    """Against a freshly truncated database with nothing seeded, every
    status the manifest cannot yet measure honestly says so rather than
    reporting a zero that could be mistaken for "zero recall.\""""
    manifest = await build_manifest(db_session)

    assert manifest["workspace_count"] == 0
    assert manifest["document_count"] == 0
    assert manifest["naive_chunk_count"] == 0
    assert manifest["structure_chunk_count"] == 0
    assert manifest["retrieval_metrics_status"] == NOT_YET_RUN
    assert manifest["red_team_status"] == NOT_YET_RUN
    # evalset/questions.yaml is a committed file, not a database row, so
    # it is still real and present even against an empty database.
    assert manifest["eval_question_count"] == 45


async def test_build_manifest_counts_seeded_rows_and_scopes_chunks_by_strategy(
    db_session: AsyncSession,
) -> None:
    workspace = Workspace(name="Test Workspace")
    db_session.add(workspace)
    await db_session.flush()

    document = Document(
        workspace_id=workspace.id,
        filename="doc.pdf",
        sha256="a" * 64,
        page_count=1,
        extraction_method=ExtractionMethod.TEXT,
    )
    db_session.add(document)
    await db_session.flush()

    db_session.add(
        Chunk(
            document_id=document.id,
            workspace_id=workspace.id,
            strategy=ChunkStrategy.NAIVE,
            text="naive chunk",
            page_start=1,
            page_end=1,
            char_start=0,
            char_end=11,
        )
    )
    db_session.add(
        Chunk(
            document_id=document.id,
            workspace_id=workspace.id,
            strategy=ChunkStrategy.STRUCTURE,
            text="structure chunk one",
            page_start=1,
            page_end=1,
            char_start=0,
            char_end=19,
        )
    )
    db_session.add(
        Chunk(
            document_id=document.id,
            workspace_id=workspace.id,
            strategy=ChunkStrategy.STRUCTURE,
            text="structure chunk two",
            page_start=1,
            page_end=1,
            char_start=19,
            char_end=38,
        )
    )
    await db_session.flush()

    manifest = await build_manifest(db_session)

    assert manifest["workspace_count"] == 1
    assert manifest["document_count"] == 1
    assert manifest["naive_chunk_count"] == 1
    assert manifest["structure_chunk_count"] == 2


async def test_build_manifest_reports_computed_once_an_eval_run_exists(
    db_session: AsyncSession,
) -> None:
    workspace = Workspace(name="Test Workspace")
    db_session.add(workspace)
    await db_session.flush()

    db_session.add(
        EvalRun(
            id=new_id(),
            embedding_backend="tfidf",
            rerank_backend="lexical",
            config_label="naive+no_rerank",
            workspace_id=workspace.id,
            category="all",
            recall_at_3=0.5,
            recall_at_5=0.6,
            precision_at_5=0.4,
            mrr=0.5,
            n_questions=10,
        )
    )
    await db_session.flush()

    manifest = await build_manifest(db_session)

    assert manifest["retrieval_metrics_status"] == "computed"
    # A real eval run existing says nothing about whether the red team
    # suite has, so that status must not flip along with it.
    assert manifest["red_team_status"] == NOT_YET_RUN
