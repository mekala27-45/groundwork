"""Engine and session setup.

Alembic migrations run synchronously (packages/api/alembic/env.py uses
psycopg3 directly), which is the ordinary pattern even for an async app,
because a migration script has no request to serve concurrently with. The
FastAPI app itself talks to Postgres through an async engine over asyncpg.
Both point at the same database; only the driver in the URL differs.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from groundwork_core.config import Settings, get_settings


def to_async_url(database_url: str) -> str:
    if "+psycopg" in database_url:
        return database_url.replace("+psycopg", "+asyncpg")
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return database_url


def to_sync_url(database_url: str) -> str:
    if "+asyncpg" in database_url:
        return database_url.replace("+asyncpg", "+psycopg")
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_url


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    global _engine
    if _engine is None:
        settings = settings or get_settings()
        _engine = create_async_engine(to_async_url(settings.database_url), pool_pre_ping=True)
    return _engine


def get_session_factory(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(settings), expire_on_commit=False)
    return _session_factory


@asynccontextmanager
async def session_scope(settings: Settings | None = None) -> AsyncIterator[AsyncSession]:
    factory = get_session_factory(settings)
    async with factory() as session:
        yield session


async def dispose_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None
