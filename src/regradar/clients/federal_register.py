"""Client for the Federal Register API.

Wraps the public Federal Register documents endpoint, returning
validated Pydantic models instead of raw JSON. No API key required.

API docs: https://www.federalregister.gov/developers/documentation/api/v1
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

import httpx
from bs4 import BeautifulSoup

from regradar.models import DocumentSearchResponse

# Base URL for the Federal Register API v1.
_BASE_URL = "https://www.federalregister.gov/api/v1"

# Fields we ask the API to return. Requesting only what we use keeps
# responses small and fast.
_REQUESTED_FIELDS = [
    "document_number",
    "title",
    "type",
    "abstract",
    "publication_date",
    "html_url",
    "comments_close_on",
    "agencies",
    "raw_text_url",
]

# Matches control characters that PostgreSQL text columns reject or that
# corrupt downstream processing: null bytes and other C0 control chars,
# except the whitespace we want to keep (tab, newline, carriage return).
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean_text(text: str) -> str:
    """Remove control characters that break storage or processing.

    Federal Register raw text occasionally contains stray null bytes and
    other control characters (artifacts of PDF-to-text conversion).
    These are stripped while preserving tabs, newlines, and carriage
    returns.
    """
    return _CONTROL_CHARS.sub("", text)


class FederalRegisterClient:
    """Async client for fetching documents from the Federal Register.

    Uses a shared httpx.AsyncClient for connection pooling. Intended to
    be used as an async context manager so the underlying connections
    are cleaned up properly:

        async with FederalRegisterClient() as client:
            response = await client.search_documents(per_page=5)
    """

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self._client = httpx.AsyncClient(
            base_url=_BASE_URL,
            timeout=timeout_seconds,
            headers={"User-Agent": "Regradar/0.1 (https://regradar.app)"},
        )

    async def __aenter__(self) -> FederalRegisterClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP connections."""
        await self._client.aclose()

    async def search_documents(
        self,
        *,
        per_page: int = 20,
        published_on_or_after: date | None = None,
        document_types: list[str] | None = None,
    ) -> DocumentSearchResponse:
        """Search recent Federal Register documents.

        Args:
            per_page: How many documents to return (max 1000, but keep small).
            published_on_or_after: Only return documents published on or
                after this date. If None, the API returns the most recent.
            document_types: Filter by type, e.g. ["RULE", "PRORULE"].
                RULE = final rule, PRORULE = proposed rule, NOTICE = notice.

        Returns:
            A validated DocumentSearchResponse containing the total count
            and the list of documents.

        Raises:
            httpx.HTTPStatusError: If the API returns a 4xx or 5xx status.
        """
        params: dict[str, Any] = {
            "per_page": per_page,
            "order": "newest",
            "fields[]": _REQUESTED_FIELDS,
        }

        if published_on_or_after is not None:
            params["conditions[publication_date][gte]"] = published_on_or_after.isoformat()

        if document_types is not None:
            params["conditions[type][]"] = document_types

        response = await self._client.get("/documents.json", params=params)
        response.raise_for_status()

        return DocumentSearchResponse.model_validate(response.json())

    async def fetch_document_text(self, raw_text_url: str) -> str:
        """Fetch the full plain text of a document.

        The raw_text_url returns lightly HTML-wrapped content, so we
        strip markup and return clean text.

        Args:
            raw_text_url: The document's raw_text_url from the API.

        Returns:
            The document's full text with HTML removed.

        Raises:
            httpx.HTTPStatusError: If the fetch fails.
        """
        response = await self._client.get(raw_text_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return _clean_text(soup.get_text())
