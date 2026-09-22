"""groundwork_api.routers.health tests: the liveness/readiness split a
container platform like Fly.io actually probes. healthz never touches the
database; readyz does, in both directions, a real SELECT 1 that succeeds
against the test database and a real connection failure against one that
cannot be reached, not just the happy path.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from groundwork_api.app import create_app
from groundwork_api.deps import get_session

pytestmark = pytest.mark.requires_postgres


async def test_healthz_reports_ok_without_touching_the_database(client: AsyncClient) -> None:
    response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readyz_reports_ok_against_a_real_database(client: AsyncClient) -> None:
    response = await client.get("/readyz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_readyz_reports_503_when_the_database_is_unreachable() -> None:
    """Its own app and override, not the shared client fixture: the point
    of this test is a session that genuinely cannot reach a database, so
    it builds one pointed at a port nothing is listening on rather than
    reusing db_session, which is a real, reachable connection."""
    app = create_app()

    async def _unreachable_session() -> AsyncIterator[AsyncSession]:
        engine = create_async_engine("postgresql+asyncpg://postgres:x@localhost:59999/nope")
        try:
            async with AsyncSession(engine) as session:
                yield session
        finally:
            await engine.dispose()

    app.dependency_overrides[get_session] = _unreachable_session
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as broken_client:
        response = await broken_client.get("/readyz")

    assert response.status_code == 503
