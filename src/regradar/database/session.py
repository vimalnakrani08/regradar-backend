"""Database engine and session management.

Provides a single shared async engine and a session factory. Other code
acquires a session via the get_session() async context manager, which
guarantees the session is committed on success and rolled back on error.
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

from regradar.config import get_settings

# A single engine is created once and shared. The engine manages a pool
# of connections — creating one per request would be slow and wasteful.
_engine: AsyncEngine = create_async_engine(
    get_settings().database_url,
    echo=False,  # set True to log all SQL (useful when debugging)
    pool_pre_ping=True,  # check connections are alive before using them
)

# Session factory bound to our engine. Each call produces a new session.
_session_factory = async_sessionmaker(
    bind=_engine,
    expire_on_commit=False,  # keep objects usable after commit
    autoflush=False,
)


def get_engine() -> AsyncEngine:
    """Return the shared async engine."""
    return _engine


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield a database session, committing on success, rolling back on error.

    Usage:
        async with get_session() as session:
            session.add(some_object)
            # auto-commits when the block exits cleanly
    """
    session = _session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
