"""Read only: the eval dashboard's own data, EvalRun and RedTeamResult
rows already persisted by scripts/run_eval.py (build order step 20).
Deliberately public, no admin token, matching section 16's definition of
done: the eval dashboard is reachable in one click, no key required, the
same as chat. Nothing here computes a number; it only ever selects rows
someone already wrote, the same "a manifest built by querying stored
records" discipline claims.py uses for README.md and RESULTS.md.

Both endpoints default to the latest run_id, reusing claims.py's own
latest_eval_run_id/latest_red_team_run_id helpers rather than
re-deriving the same query a second way. Without this, build order step
21's web app work found scripts/run_eval.py had already been run twice
against this database (once before the turn_id column below existed,
once after), leaving red_team_result holding 58 rows for what is really
29 cases: the exact duplicate-row bug run_id was added to fix in the
first place (see EvalRun.run_id's own docstring), just reappearing here
because this router, unlike claims.py, had never been scoped to it.
eval_run itself is workspace scoped and gets cleared by
scripts/seed_demo_workspaces.py's cascade delete on every reseed, so it
rarely shows this in practice, but nothing in the schema guarantees
that, and scoping both endpoints the same way costs nothing when only
one run exists.

An explicit ?run_id= still returns that exact run's rows, unfiltered by
latest, so a specific historical run stays inspectable rather than
permanently hidden the moment a newer one exists.

GET /eval/faithfulness is the one exception to "nothing here computes a
number": the faithfulness scorecard was never a stored row anywhere,
only something claims.py computes at RESULTS.md render time by iterating
every Turn's claims (build order step 23's web app work found no way for
the eval page to get this number at all otherwise, since there is no
endpoint that lists Turn rows in bulk, by design, see routers/turns'
absence). Reuses claims.py's own faithfulness_claim_counts against the
same query, so the web page and RESULTS.md can never disagree about
what counts as entailed, contradicted, or unsupported. Not scoped to a
run_id, matching build_manifest()'s own reasoning: Turn has no run_id
column at all.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from groundwork_api.claims import (
    faithfulness_claim_counts,
    latest_eval_run_id,
    latest_red_team_run_id,
)
from groundwork_api.deps import get_session
from groundwork_api.models import EvalRun, RedTeamResult, Turn
from groundwork_api.schemas import EvalRunOut, FaithfulnessScorecardOut, RedTeamResultOut

router = APIRouter(prefix="/eval", tags=["eval"])


@router.get("/runs", response_model=list[EvalRunOut])
async def list_eval_runs(
    run_id: UUID | None = None, session: AsyncSession = Depends(get_session)
) -> list[EvalRun]:
    target_run_id = run_id if run_id is not None else await latest_eval_run_id(session)
    statement = select(EvalRun).order_by(col(EvalRun.run_at).desc())
    if target_run_id is not None:
        statement = statement.where(col(EvalRun.run_id) == target_run_id)
    result = await session.execute(statement)
    return list(result.scalars())


@router.get("/red-team-results", response_model=list[RedTeamResultOut])
async def list_red_team_results(
    run_id: UUID | None = None, session: AsyncSession = Depends(get_session)
) -> list[RedTeamResult]:
    target_run_id = run_id if run_id is not None else await latest_red_team_run_id(session)
    statement = select(RedTeamResult).order_by(col(RedTeamResult.run_at).desc())
    if target_run_id is not None:
        statement = statement.where(col(RedTeamResult.run_id) == target_run_id)
    result = await session.execute(statement)
    return list(result.scalars())


@router.get("/faithfulness", response_model=FaithfulnessScorecardOut)
async def get_faithfulness_scorecard(
    session: AsyncSession = Depends(get_session),
) -> FaithfulnessScorecardOut:
    turns = list((await session.execute(select(Turn))).scalars().all())
    counts = faithfulness_claim_counts(turns)
    return FaithfulnessScorecardOut(
        entailed_count=counts["entailed"],
        contradicted_count=counts["contradicted"],
        unsupported_count=counts["unsupported"],
        total_count=sum(counts.values()),
    )
