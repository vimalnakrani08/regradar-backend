"""Pydantic models representing Federal Register data.

These models validate and structure the raw JSON returned by the
Federal Register API, giving us type-safe objects to work with
throughout the application.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, HttpUrl


class Agency(BaseModel):
    """A government agency that issued a document."""

    name: str


class FederalRegisterDocument(BaseModel):
    """A single document published in the Federal Register.

    Maps to one entry returned by the Federal Register documents API.
    Only fields we actually use are modeled; the API returns many more.
    """

    document_number: str = Field(..., description="Unique FR identifier, e.g. '2026-12345'")
    title: str
    document_type: str = Field(..., alias="type")
    abstract: str | None = Field(default=None, description="Short summary, if provided")
    publication_date: date
    html_url: HttpUrl
    comments_close_on: date | None = Field(
        default=None, description="Public comment deadline, if applicable"
    )
    agencies: list[Agency] = Field(default_factory=list)

    model_config = {"populate_by_name": True}

    @property
    def agency_names(self) -> list[str]:
        """Convenience accessor for just the agency names."""
        return [agency.name for agency in self.agencies]


class DocumentSearchResponse(BaseModel):
    """The top-level response from the FR documents search endpoint."""

    count: int = Field(..., description="Total documents matching the query")
    results: list[FederalRegisterDocument] = Field(default_factory=list)