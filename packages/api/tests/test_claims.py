from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from groundwork_api.claims import NOT_YET_RUN, build_manifest, render_template
from groundwork_api.models import (
    Chunk,
    ChunkStrategy,
    Document,
    EvalRun,
    ExtractionMethod,
    RedTeamResult,
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
    # No EvalRun or RedTeamResult row means no run_id to look up either,
    # the None branch _latest_eval_run_id/_latest_red_team_run_id take;
    # the tables built from them still render something readable rather
    # than an empty header.
    assert "No EvalRun rows yet" in manifest["retrieval_metrics_table"]
    assert manifest["injection_pass_count"] == 0
    assert manifest["injection_total_count"] == 0


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
            run_id=new_id(),
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


async def test_build_manifest_scopes_retrieval_table_to_the_latest_eval_run(
    db_session: AsyncSession,
) -> None:
    """The exact regression this project found for real: scripts/run_eval.py
    is safe to rerun (it appends fresh rows rather than overwriting, per
    its own module docstring), so a second invocation against the same
    database leaves a second run's worth of EvalRun rows sitting in the
    table alongside the first. Before EvalRun.run_id existed, nothing
    told those two runs apart, and the published retrieval table would
    have shown both stacked together as if they were one. An older run
    and a newer run are seeded here with different n_questions so the
    assertion can tell which one actually won; only the newer run's row
    should ever reach the manifest.
    """
    workspace = Workspace(name="Test Workspace")
    db_session.add(workspace)
    await db_session.flush()

    db_session.add(
        EvalRun(
            run_id=new_id(),
            run_at=datetime(2026, 1, 1, tzinfo=UTC),
            embedding_backend="tfidf",
            rerank_backend="lexical",
            config_label="naive+no_rerank",
            workspace_id=workspace.id,
            category="all",
            recall_at_3=0.1,
            recall_at_5=0.1,
            precision_at_5=0.1,
            mrr=0.1,
            n_questions=999,
        )
    )
    db_session.add(
        EvalRun(
            run_id=new_id(),
            run_at=datetime(2026, 1, 2, tzinfo=UTC),
            embedding_backend="tfidf",
            rerank_backend="lexical",
            config_label="naive+no_rerank",
            workspace_id=workspace.id,
            category="all",
            recall_at_3=0.8,
            recall_at_5=0.9,
            precision_at_5=0.4,
            mrr=0.75,
            n_questions=45,
        )
    )
    await db_session.flush()

    manifest = await build_manifest(db_session)

    table_lines = manifest["retrieval_metrics_table"].splitlines()
    assert len(table_lines) == 3, "one header row, one separator row, one data row, not two"
    assert table_lines[2].endswith("| 45 |")
    assert "999" not in manifest["retrieval_metrics_table"]
    assert manifest["boundary_comparison_table"] == "_No boundary category EvalRun rows yet._"


async def test_build_manifest_scopes_red_team_counts_to_the_latest_run(
    db_session: AsyncSession,
) -> None:
    """The same regression as the retrieval table test above, against
    RedTeamResult instead: this is the exact table where the bug was
    first found in this project, two full scripts/run_eval.py invocations
    against the same database leaving 58 rows where a single real run
    produces 29, silently doubling out_of_scope_total_count and every
    other red team figure a reader would compute from the table. An
    older run's failing out_of_scope case and a newer run's passing
    injection case are seeded here; only the newer run's rows should
    ever reach the manifest.
    """
    db_session.add(
        RedTeamResult(
            run_id=new_id(),
            run_at=datetime(2026, 1, 1, tzinfo=UTC),
            suite="out_of_scope",
            case_id="naive: a question from a stale run",
            passed=False,
            detail="stale run, must not be counted or shown",
        )
    )
    db_session.add(
        RedTeamResult(
            run_id=new_id(),
            run_at=datetime(2026, 1, 2, tzinfo=UTC),
            suite="injection",
            case_id="naive: a question from the current run",
            passed=True,
            detail=None,
        )
    )
    await db_session.flush()

    manifest = await build_manifest(db_session)

    assert manifest["injection_pass_count"] == 1
    assert manifest["injection_total_count"] == 1
    # The stale run's out_of_scope failure belongs to a run_id that is no
    # longer the latest, so it must not appear in either the count or the
    # rendered failures table, even though it is still a row in the table.
    assert manifest["out_of_scope_pass_count"] == 0
    assert manifest["out_of_scope_total_count"] == 0
    assert "stale run" not in manifest["out_of_scope_failures_table"]
    assert (
        manifest["out_of_scope_failures_table"]
        == "_No failing out_of_scope cases under naive chunking in the current run._"
    )
