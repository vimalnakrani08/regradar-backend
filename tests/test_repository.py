"""Tests for the DocumentRepository.

These run against the real local PostgreSQL database. The `session`
fixture (defined in conftest.py) wraps each test in a transaction that
is rolled back at the end, so the database is left untouched. This
tests real SQL behavior (including the upsert) rather than mocking it.
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from regradar.database.repository import DocumentRepository
from regradar.models import FederalRegisterDocument


def _sample_doc(number: str = "TEST-0001") -> FederalRegisterDocument:
    """Build a sample API document for testing."""
    return FederalRegisterDocument.model_validate(
        {
            "document_number": number,
            "title": "Test Proposed Rule",
            "type": "Proposed Rule",
            "abstract": "A test abstract.",
            "publication_date": "2026-06-02",
            "html_url": "https://www.federalregister.gov/documents/test",
            "comments_close_on": "2026-08-01",
            "agencies": [{"name": "Test Agency"}],
        }
    )


@pytest.mark.asyncio
async def test_upsert_and_get(session: AsyncSession) -> None:
    """Storing a document then fetching it returns the same data."""
    repo = DocumentRepository(session)

    count = await repo.upsert_documents([_sample_doc()])
    assert count == 1

    stored = await repo.get_document("TEST-0001")
    assert stored is not None
    assert stored.title == "Test Proposed Rule"
    assert stored.agency_names == "Test Agency"
    assert stored.comments_close_on == date(2026, 8, 1)


@pytest.mark.asyncio
async def test_document_exists(session: AsyncSession) -> None:
    """document_exists reflects whether a document is stored."""
    repo = DocumentRepository(session)

    assert await repo.document_exists("TEST-0001") is False
    await repo.upsert_documents([_sample_doc()])
    assert await repo.document_exists("TEST-0001") is True


@pytest.mark.asyncio
async def test_upsert_is_idempotent(session: AsyncSession) -> None:
    """Upserting the same document twice updates rather than duplicates."""
    repo = DocumentRepository(session)

    await repo.upsert_documents([_sample_doc()])
    updated = _sample_doc()
    updated.title = "Updated Title"
    await repo.upsert_documents([updated])

    stored = await repo.get_document("TEST-0001")
    assert stored is not None
    assert stored.title == "Updated Title"
