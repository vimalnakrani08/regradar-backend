"""The ingestion pipeline: fetch -> chunk -> embed -> store.

Orchestrates the full flow of bringing Federal Register documents into
the database with embedded chunks, ready for semantic search. Designed
to be idempotent: re-running never duplicates data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from regradar.chunking import chunk_document
from regradar.clients.federal_register import FederalRegisterClient
from regradar.database.repository import DocumentRepository
from regradar.database.session import get_session
from regradar.embeddings.base import Embedder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    """Summary of one ingestion run."""

    fetched: int
    skipped_existing: int
    ingested: int
    chunks_stored: int


async def ingest_recent_documents(
    embedder: Embedder,
    *,
    per_page: int = 20,
    document_types: list[str] | None = None,
) -> IngestionResult:
    """Fetch recent FR documents and ingest new ones with embeddings.

    Args:
        embedder: Any Embedder implementation (injected dependency).
        per_page: How many recent documents to fetch.
        document_types: Optional FR type filter, e.g. ["PRORULE", "RULE"].

    Returns:
        Counts summarizing what happened.
    """
    fetched = skipped = ingested = chunks_total = 0

    async with FederalRegisterClient() as client:
        response = await client.search_documents(per_page=per_page, document_types=document_types)
        fetched = len(response.results)

        for doc in response.results:
            async with get_session() as session:
                repo = DocumentRepository(session)

                if await repo.document_exists(doc.document_number):
                    skipped += 1
                    continue

                await repo.upsert_documents([doc])

                if doc.raw_text_url is None:
                    logger.info("No raw text for %s; stored metadata only", doc.document_number)
                    ingested += 1
                    continue

                text = await client.fetch_document_text(str(doc.raw_text_url))
                chunks = chunk_document(text)
                if not chunks:
                    ingested += 1
                    continue

                vectors = await embedder.embed([c.content for c in chunks])

                stored = await repo.replace_chunks(
                    doc.document_number,
                    [
                        (c.chunk_index, c.section, c.content, vector)
                        for c, vector in zip(chunks, vectors, strict=True)
                    ],
                )
                chunks_total += stored
                ingested += 1
                logger.info("Ingested %s with %d chunks", doc.document_number, stored)

    return IngestionResult(
        fetched=fetched,
        skipped_existing=skipped,
        ingested=ingested,
        chunks_stored=chunks_total,
    )
