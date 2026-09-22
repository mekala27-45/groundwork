"""Repository-wide pytest configuration.

Two things live here rather than in a per-package conftest, deliberately,
because day 1's note says to run the full suite rather than the new file:
a conftest.py placed in a new tests/ directory once shadowed the per
package conftests and broke collection of a whole module, and it passed
when that one file was run alone. Centralizing the Postgres fixtures here
avoids ever repeating that mistake, since there is only one conftest for
pytest to resolve instead of one per package that could shadow another.

requires_postgres: a real TCP probe of localhost:5432, cached for the
session. Tests marked @pytest.mark.requires_postgres are auto-skipped with
a named reason when it is unreachable, per the carried-forward rule that
any test which shells out (or, here, reaches out to a real service) gets
an availability probe rather than a bare skip or a silent failure. CI
(.github/workflows/ci.yml) starts a real Postgres service container, so
this only ever skips locally on a machine with no database running.
"""

from __future__ import annotations

import socket
from collections.abc import AsyncIterator
from functools import lru_cache

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel, text

from groundwork_api.app import create_app
from groundwork_api.db import to_async_url
from groundwork_api.deps import get_session
from groundwork_core.config import Settings

TEST_DATABASE_URL = (
    "postgresql+psycopg://postgres:groundwork_dev_password@localhost:5432/groundwork_test"
)


@lru_cache(maxsize=1)
def postgres_available(host: str = "localhost", port: int = 5432, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "requires_postgres: needs a real Postgres with pgvector (see conftest.py)"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if postgres_available():
        return
    skip_marker = pytest.mark.skip(
        reason="no Postgres reachable at localhost:5432 (run `make db-up` first)"
    )
    for item in items:
        if "requires_postgres" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    return Settings(database_url=TEST_DATABASE_URL)


@pytest_asyncio.fixture
async def db_session(test_settings: Settings) -> AsyncIterator[AsyncSession]:
    """A fresh engine per test, not the process-wide singleton in
    groundwork_api.db.

    pytest-asyncio hands each test function its own event loop, and
    asyncpg connections are bound to the loop that created them. The
    application's get_engine()/get_session_factory() cache a single engine
    for the process's one long-lived loop, which is correct for a running
    FastAPI app and wrong for a test suite: reusing that cached engine
    across two tests with two different loops raises "attached to a
    different loop", which this fixture avoids by never touching that
    cache at all.
    """
    engine = create_async_engine(to_async_url(test_settings.database_url), pool_pre_ping=True)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with factory() as session:
            yield session
            await session.rollback()
        # Truncate rather than rely solely on rollback, so a test that
        # commits partway through (several of the isolation tests do,
        # deliberately, to prove cross-request behavior) still leaves a
        # clean table for the next test.
        async with engine.begin() as conn:
            table_names = ", ".join(
                f'"{t.name}"' for t in reversed(SQLModel.metadata.sorted_tables)
            )
            await conn.execute(text(f"TRUNCATE TABLE {table_names} RESTART IDENTITY CASCADE"))
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncIterator[AsyncClient]:
    """An httpx client wired directly to a fresh groundwork_api app (one
    per test, per app.py's own module docstring), with get_session
    overridden to hand out this test's own db_session rather than the
    app's process wide engine singleton.

    httpx.ASGITransport, not FastAPI's TestClient: TestClient drives the
    app from a background anyio portal with its own event loop, and
    db_session's asyncpg connection is bound to pytest-asyncio's loop for
    this test function. Crossing loops is exactly the "attached to a
    different loop" failure db_session's own docstring warns about;
    ASGITransport runs requests on the caller's loop, this test's loop, so
    the override never crosses one.

    The app's lifespan (configure_logging, dispose_engine) deliberately
    does not run here: dispose_engine only tears down the get_session_factory
    singleton this fixture's override bypasses entirely, and structlog logs
    fine, just unformatted, without configure_logging having run.
    """

    app = create_app()

    async def _override_get_session() -> AsyncIterator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = _override_get_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as async_client:
        yield async_client
