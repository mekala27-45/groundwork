"""Read only: the eval dashboard's own data, EvalRun and RedTeamResult
rows already persisted by scripts/run_eval.py (build order step 20).
Deliberately public, no admin token, matching section 16's definition of
done: the eval dashboard is reachable in one click, no key required, the
same as chat. Nothing here computes a number; it only ever selects rows
someone already wrote, the same "a manifest built by querying stored
records" discipline claims.py uses for README.md and RESULTS.md.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from groundwork_api.deps import get_session
from groundwork_api.models import EvalRun, RedTeamResult
from groundwork_api.schemas import EvalRunOut, RedTeamResultOut

router = APIRouter(prefix="/eval", tags=["eval"])


@router.get("/runs", response_model=list[EvalRunOut])
async def list_eval_runs(session: AsyncSession = Depends(get_session)) -> list[EvalRun]:
    result = await session.execute(select(EvalRun).order_by(col(EvalRun.run_at).desc()))
    return list(result.scalars())


@router.get("/red-team-results", response_model=list[RedTeamResultOut])
async def list_red_team_results(
    session: AsyncSession = Depends(get_session),
) -> list[RedTeamResult]:
    result = await session.execute(select(RedTeamResult).order_by(col(RedTeamResult.run_at).desc()))
    return list(result.scalars())
