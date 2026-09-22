"""groundwork_api.db tests: the URL driver rewrites (pure functions, no
database needed) and the process wide engine/session factory singleton,
cached across calls and reset by dispose_engine, the behavior
conftest.py's own db_session fixture deliberately avoids exercising since
it builds an isolated engine per test instead. Nothing else in this test
suite calls get_engine, get_session_factory, or dispose_engine directly
(every router test overrides groundwork_api.deps.get_session instead), so
this file is the only place that caching contract is actually proven.
"""

from __future__ import annotations

import pytest

from groundwork_api.db import (
    dispose_engine,
    get_engine,
    get_session_factory,
    to_async_url,
    to_sync_url,
)
from groundwork_core.config import Settings

pytestmark = pytest.mark.requires_postgres


def test_to_async_url_rewrites_psycopg_to_asyncpg() -> None:
    assert to_async_url("postgresql+psycopg://u:p@host/db") == "postgresql+asyncpg://u:p@host/db"


def test_to_async_url_adds_asyncpg_to_a_bare_postgresql_url() -> None:
    assert to_async_url("postgresql://u:p@host/db") == "postgresql+asyncpg://u:p@host/db"


def test_to_async_url_leaves_an_already_async_url_unchanged() -> None:
    assert to_async_url("postgresql+asyncpg://u:p@host/db") == "postgresql+asyncpg://u:p@host/db"


def test_to_sync_url_rewrites_asyncpg_to_psycopg() -> None:
    assert to_sync_url("postgresql+asyncpg://u:p@host/db") == "postgresql+psycopg://u:p@host/db"


def test_to_sync_url_adds_psycopg_to_a_bare_postgresql_url() -> None:
    assert to_sync_url("postgresql://u:p@host/db") == "postgresql+psycopg://u:p@host/db"


def test_to_sync_url_leaves_an_already_sync_url_unchanged() -> None:
    assert to_sync_url("postgresql+psycopg://u:p@host/db") == "postgresql+psycopg://u:p@host/db"


async def test_get_engine_and_get_session_factory_are_cached_and_dispose_engine_resets_them() -> (
    None
):
    """create_async_engine builds a connection pool lazily, no socket ever
    opens until a query runs, so this proves the caching contract itself
    without needing a reachable database. requires_postgres is kept
    anyway since dispose_engine's real deployment purpose is tied to a
    real database lifecycle, and this environment always has one."""
    await dispose_engine()  # start from a known, uncached state
    try:
        settings = Settings(
            database_url="postgresql+psycopg://postgres:x@localhost:5432/groundwork_dev"
        )

        first_engine = get_engine(settings)
        assert get_engine() is first_engine

        first_factory = get_session_factory()
        assert get_session_factory() is first_factory

        await dispose_engine()

        second_engine = get_engine(settings)
        assert second_engine is not first_engine
        await dispose_engine()
    finally:
        await dispose_engine()
