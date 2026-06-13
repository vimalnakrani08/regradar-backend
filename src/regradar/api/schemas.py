"""API request/response models (DTOs).

These are deliberately separate from the ORM models (`database.models`) and
the domain models (`models`, `generation.base`). The wire contract the Next.js
frontend depends on should be free to evolve independently of the database
schema, and ORM objects should never be serialized directly to clients.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

# Chunk content can be long; cap what we echo back to clients so responses
# stay lean. The full document is always available via the documents endpoint.
_EXCERPT_CHARS = 600


class SourceOut(BaseModel):
    """A retrieved chunk returned to the client, as a citation or search hit."""

    document_number: str = Field(..., description="FR document number, e.g. '2026-12345'.")
    section: str | None = Field(default=None, description="Structural section, e.g. 'SUMMARY'.")
    similarity: float = Field(..., description="Cosine similarity to the query (1.0 = identical).")
    excerpt: str = Field(..., description="The chunk's text, truncated for display.")


class AskRequest(BaseModel):
    """A question to answer over the indexed corpus."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The user's natural-language question.",
    )


class AskResponse(BaseModel):
    """A generated answer plus its citations.

    `has_answer` is False when retrieval found nothing relevant enough; the
    client should render `answer` (an honest disclaimer) and expect no sources.
    """

    answer: str = Field(..., description="The answer, or an honest 'no information' disclaimer.")
    has_answer: bool = Field(..., description="True if grounded in retrieved sources.")
    sources: list[SourceOut] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Ranked semantic-search results for a query."""

    query: str
    results: list[SourceOut] = Field(default_factory=list)


class DocumentOut(BaseModel):
    """A Federal Register document's metadata for list and detail views."""

    document_number: str
    title: str
    document_type: str
    abstract: str | None = None
    publication_date: date
    html_url: str
    comments_close_on: date | None = None
    agency_names: list[str] = Field(default_factory=list)


class DocumentListResponse(BaseModel):
    """A page of documents plus pagination metadata."""

    count: int = Field(..., description="Total documents stored (not just this page).")
    limit: int
    offset: int
    results: list[DocumentOut] = Field(default_factory=list)


def excerpt(content: str) -> str:
    """Truncate chunk content to a display-friendly length."""
    if len(content) <= _EXCERPT_CHARS:
        return content
    return content[:_EXCERPT_CHARS].rstrip() + "…"
