"""Dependency-injection providers and the auth / rate-limit seams.

Routers declare what they need via `Depends(...)`; this module wires it up.
Sessions are per-request; the embedder and answer generator are process-wide
singletons created at startup and read off `app.state` (they wrap reusable
HTTP clients, so rebuilding them per request would be wasteful).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from regradar.database.repository import DocumentRepository
from regradar.database.session import get_session
from regradar.embeddings.base import Embedder
from regradar.generation.base import AnswerGenerator
from regradar.generation.rag import RagService
from regradar.search import SemanticSearch


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Yield a per-request database session (committed/rolled back on exit)."""
    async with get_session() as session:
        yield session


def get_repository(session: AsyncSession = Depends(get_db_session)) -> DocumentRepository:
    """Build a repository bound to the request's session."""
    return DocumentRepository(session)


def get_embedder(request: Request) -> Embedder:
    """Return the shared embedder singleton from app state."""
    embedder: Embedder = request.app.state.embedder
    return embedder


def get_generator(request: Request) -> AnswerGenerator:
    """Return the shared answer-generator singleton from app state."""
    generator: AnswerGenerator = request.app.state.generator
    return generator


def get_rag_service(
    embedder: Embedder = Depends(get_embedder),
    repository: DocumentRepository = Depends(get_repository),
    generator: AnswerGenerator = Depends(get_generator),
) -> RagService:
    """Assemble the RAG service for one request."""
    return RagService(embedder, repository, generator)


def get_search_service(
    embedder: Embedder = Depends(get_embedder),
    repository: DocumentRepository = Depends(get_repository),
) -> SemanticSearch:
    """Assemble the semantic-search service for one request."""
    return SemanticSearch(embedder, repository)


async def require_auth(request: Request) -> None:
    """Authentication seam — currently a no-op pass-through.

    Wired into protected routers now so that turning on real auth later (API
    key or JWT) is a one-place change here, with no router edits. Until then
    every request is allowed; this intentionally does not reject anything.
    """
    return None


async def enforce_rate_limit(request: Request) -> None:
    """Rate-limiting seam — currently a no-op pass-through.

    Applied to the expensive endpoints (notably /ask, which spends on the
    Anthropic API) so a real limiter (e.g. slowapi backed by Redis, keyed by
    client identity from `require_auth`) can be dropped in here later without
    touching the routes.
    """
    return None
