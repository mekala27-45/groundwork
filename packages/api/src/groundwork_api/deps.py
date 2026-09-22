"""FastAPI dependencies shared across routers: a request scoped database
session, and the admin token gate config.py's own Settings.admin_token
docstring describes (DELETE /workspaces/{id} only, fails closed with no
token configured, never required for chat, upload, or reading the eval
dashboard).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from groundwork_api.db import get_session_factory
from groundwork_core.config import get_settings


async def get_session() -> AsyncIterator[AsyncSession]:
    factory = get_session_factory()
    async with factory() as session:
        yield session


async def require_admin_token(x_admin_token: str | None = Header(default=None)) -> None:
    configured = get_settings().admin_token
    if not configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="admin actions are disabled: no admin token is configured",
        )
    if x_admin_token != configured:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid admin token")
