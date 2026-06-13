"""Tests for the semantic search service.

Stub embedder and repository keep these offline and deterministic — no real
embeddings or database. The focus is the relevance floor: search must drop
below-threshold chunks so it never implies relevance that isn't there.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from regradar.search import SemanticSearch


class StubEmbedder:
    """Embedder returning a fixed vector regardless of input."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


def _chunk(document_number: str, content: str = "chunk text") -> SimpleNamespace:
    """Stand-in for a DocumentChunk with the fields SemanticSearch reads."""
    return SimpleNamespace(document_number=document_number, section="SUMMARY", content=content)


class StubRepository:
    """Repository whose search_chunks returns preset (chunk, similarity) pairs."""

    def __init__(self, matches: list[tuple[SimpleNamespace, float]]) -> None:
        self._matches = matches

    async def search_chunks(
        self, query_embedding: list[float], *, limit: int = 10
    ) -> list[tuple[SimpleNamespace, float]]:
        return self._matches[:limit]


@pytest.mark.asyncio
async def test_search_returns_chunks_at_or_above_threshold() -> None:
    """Chunks at or above the floor are returned; weaker ones are dropped."""
    matches = [
        (_chunk("2026-11854"), 0.65),
        (_chunk("2026-11843"), 0.45),  # exactly at the floor — kept
        (_chunk("2026-10001"), 0.30),  # below the floor — dropped
    ]
    service = SemanticSearch(StubEmbedder(), StubRepository(matches), similarity_threshold=0.45)  # type: ignore[arg-type]

    results = await service.search("prediction markets")

    assert [s.document_number for s in results] == ["2026-11854", "2026-11843"]


@pytest.mark.asyncio
async def test_search_below_floor_returns_empty() -> None:
    """An out-of-scope query whose every match is below the floor returns []."""
    # Mirrors the live "cryptocurrency staking" case: top-K all ~0.24-0.26.
    matches = [(_chunk(f"2026-1100{i}"), 0.26 - i * 0.01) for i in range(5)]
    service = SemanticSearch(StubEmbedder(), StubRepository(matches), similarity_threshold=0.45)  # type: ignore[arg-type]

    results = await service.search("cryptocurrency staking")

    assert results == []
