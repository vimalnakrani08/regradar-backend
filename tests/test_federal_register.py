"""Tests for the Federal Register API client.

These tests mock the HTTP layer with respx so they run fast, offline,
and deterministically — never hitting the live government API.
"""

from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx

from regradar.clients.federal_register import FederalRegisterClient

# A minimal but realistic sample of what the FR API returns, used to
# mock the HTTP response. Only the fields our models care about.
_SAMPLE_RESPONSE = {
    "count": 2,
    "results": [
        {
            "document_number": "2026-10969",
            "title": "Pipeline Safety: Breakout Tank Inspection Rule",
            "type": "Proposed Rule",
            "abstract": "PHMSA proposes new inspection requirements.",
            "publication_date": "2026-06-02",
            "html_url": "https://www.federalregister.gov/documents/2026/06/02/2026-10969/test",
            "comments_close_on": "2026-08-03",
            "agencies": [
                {"name": "Transportation Department"},
                {"name": "Pipeline and Hazardous Materials Safety Administration"},
            ],
        },
        {
            "document_number": "2026-10968",
            "title": "Ballot Mail for Federal Elections",
            "type": "Proposed Rule",
            "abstract": None,
            "publication_date": "2026-06-02",
            "html_url": "https://www.federalregister.gov/documents/2026/06/02/2026-10968/test",
            "comments_close_on": None,
            "agencies": [{"name": "Postal Service"}],
        },
    ],
}


@respx.mock
@pytest.mark.asyncio
async def test_search_documents_parses_response() -> None:
    """The client should parse a well-formed API response into models."""
    respx.get("https://www.federalregister.gov/api/v1/documents.json").mock(
        return_value=httpx.Response(200, json=_SAMPLE_RESPONSE)
    )

    async with FederalRegisterClient() as client:
        result = await client.search_documents(per_page=2)

    assert result.count == 2
    assert len(result.results) == 2

    first = result.results[0]
    assert first.document_number == "2026-10969"
    assert first.document_type == "Proposed Rule"
    assert first.publication_date == date(2026, 6, 2)
    assert first.comments_close_on == date(2026, 8, 3)
    assert first.agency_names == [
        "Transportation Department",
        "Pipeline and Hazardous Materials Safety Administration",
    ]


@respx.mock
@pytest.mark.asyncio
async def test_search_documents_handles_missing_optional_fields() -> None:
    """Documents without abstract or comment deadline should still parse."""
    respx.get("https://www.federalregister.gov/api/v1/documents.json").mock(
        return_value=httpx.Response(200, json=_SAMPLE_RESPONSE)
    )

    async with FederalRegisterClient() as client:
        result = await client.search_documents(per_page=2)

    second = result.results[1]
    assert second.abstract is None
    assert second.comments_close_on is None
    assert second.agency_names == ["Postal Service"]


@respx.mock
@pytest.mark.asyncio
async def test_search_documents_raises_on_server_error() -> None:
    """A 5xx response should raise, not silently return bad data."""
    respx.get("https://www.federalregister.gov/api/v1/documents.json").mock(
        return_value=httpx.Response(500)
    )

    async with FederalRegisterClient() as client:
        with pytest.raises(httpx.HTTPStatusError):
            await client.search_documents()


@respx.mock
@pytest.mark.asyncio
async def test_agency_with_only_raw_name() -> None:
    """An agency providing only raw_name should still resolve a name."""
    payload = {
        "count": 1,
        "results": [
            {
                "document_number": "2026-99999",
                "title": "Test Rule",
                "type": "Proposed Rule",
                "abstract": None,
                "publication_date": "2026-06-12",
                "html_url": "https://www.federalregister.gov/documents/test",
                "comments_close_on": None,
                "agencies": [
                    {"name": "Defense Department"},
                    {"raw_name": "Office of the Secretary"},
                ],
            }
        ],
    }
    respx.get("https://www.federalregister.gov/api/v1/documents.json").mock(
        return_value=httpx.Response(200, json=payload)
    )

    async with FederalRegisterClient() as client:
        result = await client.search_documents(per_page=1)

    assert result.results[0].agency_names == [
        "Defense Department",
        "Office of the Secretary",
    ]


def test_clean_text_strips_null_bytes() -> None:
    """Null bytes and control chars are removed; normal whitespace kept."""
    from regradar.clients.federal_register import _clean_text

    dirty = "Hello\x00World\tTabbed\nNewline\x07Bell"
    cleaned = _clean_text(dirty)

    assert "\x00" not in cleaned
    assert "\x07" not in cleaned
    assert cleaned == "HelloWorld\tTabbed\nNewlineBell"
