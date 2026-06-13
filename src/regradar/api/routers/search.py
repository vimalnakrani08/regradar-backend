"""The /search endpoint: semantic search over document chunks (no LLM)."""

from __future__ import annotations

import openai
from fastapi import APIRouter, Depends, Query

from regradar.api.dependencies import enforce_rate_limit, get_search_service, require_auth
from regradar.api.errors import APIError
from regradar.api.schemas import SearchResponse, SourceOut, excerpt
from regradar.search import SemanticSearch

router = APIRouter(
    tags=["search"],
    dependencies=[Depends(require_auth), Depends(enforce_rate_limit)],
)


@router.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=3, max_length=1000, description="Search query."),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum results to return."),
    service: SemanticSearch = Depends(get_search_service),
) -> SearchResponse:
    """Return the document chunks most semantically similar to the query."""
    try:
        sources = await service.search(q, limit=limit)
    except openai.APIError as exc:
        raise APIError(
            status_code=503,
            code="search_unavailable",
            message="The search service is temporarily unavailable. Please try again.",
        ) from exc

    return SearchResponse(
        query=q,
        results=[
            SourceOut(
                document_number=s.document_number,
                section=s.section,
                similarity=s.similarity,
                excerpt=excerpt(s.content),
            )
            for s in sources
        ],
    )
