"""groundwork_api.routers.eval tests: read only, deliberately unauthenticated
(section 16's definition of done keeps the eval dashboard reachable with
no key, same as chat), so these tests only need to prove a seeded EvalRun
or RedTeamResult row comes back through the API exactly as stored, most
recent first.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from groundwork_api.models import EvalRun, RedTeamResult, Workspace
from groundwork_core.ids import new_id

pytestmark = pytest.mark.requires_postgres


async def _seed_workspace(session: AsyncSession) -> Workspace:
    workspace = Workspace(name="eval-router-test")
    session.add(workspace)
    await session.commit()
    return workspace


async def test_list_eval_runs_is_empty_with_nothing_seeded(client: AsyncClient) -> None:
    response = await client.get("/eval/runs")
    assert response.status_code == 200
    assert response.json() == []


async def test_list_eval_runs_returns_a_seeded_run(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    workspace = await _seed_workspace(db_session)
    run = EvalRun(
        run_id=new_id(),
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
    db_session.add(run)
    await db_session.commit()

    response = await client.get("/eval/runs")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["run_id"] == str(run.run_id)
    assert body[0]["config_label"] == "naive+no_rerank"
    assert body[0]["workspace_id"] == str(workspace.id)
    assert body[0]["n_questions"] == 45


async def test_list_red_team_results_is_empty_with_nothing_seeded(client: AsyncClient) -> None:
    response = await client.get("/eval/red-team-results")
    assert response.status_code == 200
    assert response.json() == []


async def test_list_red_team_results_returns_a_seeded_row(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    result = RedTeamResult(
        run_id=new_id(), suite="injection", case_id="case-1", passed=True, detail=None
    )
    db_session.add(result)
    await db_session.commit()

    response = await client.get("/eval/red-team-results")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["run_id"] == str(result.run_id)
    assert body[0]["suite"] == "injection"
    assert body[0]["case_id"] == "case-1"
    assert body[0]["passed"] is True
    assert body[0]["detail"] is None
