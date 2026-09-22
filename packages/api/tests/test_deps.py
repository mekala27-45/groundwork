"""groundwork_api.deps.get_session tests: the real dependency, not the
override every router test installs in its place. require_admin_token's
three branches (no token configured, wrong token, correct token) are
already proven through real HTTP requests in test_workspaces_router.py's
delete tests; duplicating that here as a direct call would just repeat
the same three assertions without adding coverage, so this file's only
job is the one path those tests never touch: get_session() actually
talking to a real database through the process wide engine.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import cast

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from groundwork_api.db import dispose_engine
from groundwork_api.deps import get_session

pytestmark = pytest.mark.requires_postgres

TEST_DATABASE_URL = (
    "postgresql+psycopg://postgres:groundwork_dev_password@localhost:5432/groundwork_test"
)


async def test_get_session_yields_a_working_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_session() takes no arguments, so it always reads Settings from
    the environment through get_settings(); pointing GROUNDWORK_DATABASE_URL
    at the real test database, the same one db_session itself uses, is
    what lets this test run safely alongside the rest of the suite."""
    monkeypatch.setenv("GROUNDWORK_DATABASE_URL", TEST_DATABASE_URL)
    await dispose_engine()
    try:
        # get_session is declared -> AsyncIterator[AsyncSession] (the
        # honest, minimal contract for a FastAPI dependency), but it is
        # implemented with yield, so the object it actually returns is an
        # AsyncGenerator and does support aclose(); this call site is the
        # one place in the suite that drives it directly rather than
        # through FastAPI's own dependency injection, so it needs that.
        session_iter = cast(AsyncGenerator[AsyncSession, None], get_session())
        session = await anext(session_iter)
        try:
            result = await session.execute(text("SELECT 1"))
            assert result.scalar_one() == 1
        finally:
            await session_iter.aclose()
    finally:
        await dispose_engine()
