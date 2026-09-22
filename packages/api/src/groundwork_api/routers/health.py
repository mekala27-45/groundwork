"""Liveness and readiness, the standard split a container platform like
Fly.io actually probes: healthz answers instantly and proves nothing more
than that the process is up and serving, readyz proves the one dependency
that actually has to be reachable for any other route to work, a real
query against Postgres, not merely that a connection object exists.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from groundwork_api.deps import get_session

router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(session: AsyncSession = Depends(get_session)) -> dict[str, str]:
    try:
        await session.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="database unreachable"
        ) from exc
    return {"status": "ok"}
