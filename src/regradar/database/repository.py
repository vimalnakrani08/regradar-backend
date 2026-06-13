"""Repository for document persistence.

All database access for documents goes through this class. The rest of
the application calls these methods and never writes raw queries or
manages sessions directly. This keeps data-access logic in one place
and makes the code easy to test and change.
"""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from regradar.database.models import Document, DocumentChunk
from regradar.models import FederalRegisterDocument


class DocumentRepository:
    """Data-access methods for Federal Register documents.

    A repository instance wraps a single AsyncSession. Callers provide
    the session (dependency injection), which keeps the repository
    decoupled from how sessions are created and lets tests inject a
    test session.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def document_exists(self, document_number: str) -> bool:
        """Return True if a document with this number is already stored.

        Used by ingestion to skip documents we've already fetched,
        avoiding redundant work. Runs as a cheap indexed primary-key
        lookup.
        """
        stmt = select(Document.document_number).where(Document.document_number == document_number)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def get_document(self, document_number: str) -> Document | None:
        """Fetch a single document by its number, or None if not found."""
        return await self._session.get(Document, document_number)

    async def upsert_documents(self, documents: list[FederalRegisterDocument]) -> int:
        """Insert documents, updating any that already exist.

        Uses PostgreSQL's INSERT ... ON CONFLICT DO UPDATE so a single
        statement handles both new and existing documents efficiently,
        rather than checking-then-inserting one at a time.

        Args:
            documents: API documents to store.

        Returns:
            The number of documents processed.
        """
        if not documents:
            return 0

        rows = [
            {
                "document_number": doc.document_number,
                "title": doc.title,
                "document_type": doc.document_type,
                "abstract": doc.abstract,
                "publication_date": doc.publication_date,
                "html_url": str(doc.html_url),
                "comments_close_on": doc.comments_close_on,
                "agency_names": " | ".join(doc.agency_names),
            }
            for doc in documents
        ]

        stmt = pg_insert(Document).values(rows)
        # On conflict (same primary key), update the mutable fields.
        stmt = stmt.on_conflict_do_update(
            index_elements=["document_number"],
            set_={
                "title": stmt.excluded.title,
                "document_type": stmt.excluded.document_type,
                "abstract": stmt.excluded.abstract,
                "publication_date": stmt.excluded.publication_date,
                "html_url": stmt.excluded.html_url,
                "comments_close_on": stmt.excluded.comments_close_on,
                "agency_names": stmt.excluded.agency_names,
            },
        )

        await self._session.execute(stmt)
        return len(documents)

    async def replace_chunks(
        self,
        document_number: str,
        chunks: list[tuple[int, str | None, str, list[float]]],
    ) -> int:
        """Replace all chunks for a document with a new set.

        Deleting-then-inserting keeps re-ingestion idempotent: running
        ingestion twice never duplicates chunks.

        Args:
            document_number: The parent document.
            chunks: Tuples of (chunk_index, section, content, embedding).

        Returns:
            Number of chunks stored.
        """
        from sqlalchemy import delete

        from regradar.database.models import DocumentChunk

        await self._session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_number == document_number)
        )

        self._session.add_all(
            DocumentChunk(
                document_number=document_number,
                chunk_index=index,
                section=section,
                content=content,
                embedding=embedding,
            )
            for index, section, content, embedding in chunks
        )
        return len(chunks)
