"""Semantic search over document chunks.

A thin orchestrator: embed the query, retrieve the nearest chunks, and return
typed results. Kept separate from RagService so the search endpoint can expose
ranked chunks without ever touching the LLM. Like the other orchestrators, its
collaborators are injected rather than constructed here.
"""

from __future__ import annotations

from regradar.config import get_settings
from regradar.database.repository import DocumentRepository
from regradar.embeddings.base import Embedder
from regradar.generation.base import Source


class SemanticSearch:
    """Embeds a query and returns the most similar chunks as `Source`s."""

    def __init__(
        self,
        embedder: Embedder,
        repository: DocumentRepository,
        *,
        similarity_threshold: float | None = None,
    ) -> None:
        settings = get_settings()
        self._embedder = embedder
        self._repository = repository
        self._threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.search_similarity_threshold
        )

    async def search(self, query: str, *, limit: int = 10) -> list[Source]:
        """Return the chunks most semantically similar to the query.

        Results below the similarity threshold are dropped, not shown: search
        should never imply relevance that isn't there. A genuinely out-of-scope
        query therefore returns an empty list rather than weak, misleading
        "matches" — the same honesty principle as the answer threshold.

        Args:
            query: The natural-language search query.
            limit: Maximum number of chunks to consider (before the floor).

        Returns:
            Matching chunks at or above the threshold as `Source` objects, best
            match first (possibly fewer than `limit`, or empty). Reuses the same
            `Source` model the generation layer cites, so search results and
            answer citations share one shape.
        """
        query_vectors = await self._embedder.embed([query])
        matches = await self._repository.search_chunks(query_vectors[0], limit=limit)
        return [
            Source(
                document_number=chunk.document_number,
                section=chunk.section,
                similarity=similarity,
                content=chunk.content,
            )
            for chunk, similarity in matches
            if similarity >= self._threshold
        ]
