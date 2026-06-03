"""SQLAlchemy ORM models defining the database schema.

Two tables:
  - documents: one row per Federal Register document
  - document_chunks: many rows per document, each an embeddable text
    chunk with its vector

The relationship is one-to-many (one document -> many chunks).
"""

from __future__ import annotations

from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Date, DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from regradar.config import get_settings

_EMBEDDING_DIM = get_settings().embedding_dimension


class Base(DeclarativeBase):
    """Base class for all ORM models."""


class Document(Base):
    """A single Federal Register document."""

    __tablename__ = "documents"

    # We use the FR document_number as the natural primary key — it's
    # unique and stable, so no need for a separate surrogate id.
    document_number: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    document_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    publication_date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    html_url: Mapped[str] = mapped_column(Text, nullable=False)
    comments_close_on: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    agency_names: Mapped[str] = mapped_column(Text, nullable=False, default="")

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentChunk(Base):
    """An embeddable chunk of text from a document."""

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    document_number: Mapped[str] = mapped_column(
        ForeignKey("documents.document_number", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Position of this chunk within its document (0, 1, 2, ...).
    chunk_index: Mapped[int] = mapped_column(nullable=False)
    # Which structural section this chunk came from (e.g. "SUMMARY").
    section: Mapped[str | None] = mapped_column(String(128), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(_EMBEDDING_DIM), nullable=True)

    document: Mapped[Document] = relationship(back_populates="chunks")


# HNSW index for fast approximate nearest-neighbor search on embeddings.
# vector_cosine_ops = use cosine distance (standard for text embeddings).
Index(
    "ix_document_chunks_embedding_hnsw",
    DocumentChunk.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)
