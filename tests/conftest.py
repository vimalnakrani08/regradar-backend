"""Shared pytest fixtures.

pytest automatically discovers and loads this file. We define database
fixtures here so async tests get a session bound to the *current* test
event loop, avoiding cross-loop errors from reusing a module-level
engine.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from regradar.config import get_settings


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """Yield a session wrapped in a transaction that always rolls back.

    A fresh engine is created and disposed within this fixture so it
    binds to the test's own event loop. Each test's writes are rolled
    back, leaving the database untouched.
    """
    engine = create_async_engine(get_settings().database_url)
    connection = await engine.connect()
    transaction = await connection.begin()
    test_session = AsyncSession(bind=connection, expire_on_commit=False)
    try:
        yield test_session
    finally:
        await test_session.close()
        await transaction.rollback()
        await connection.close()
        await engine.dispose()
