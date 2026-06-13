"""The /documents endpoints: list and fetch Federal Register documents."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Path, Query

from regradar.api.dependencies import get_repository, require_auth
from regradar.api.errors import APIError
from regradar.api.schemas import DocumentListResponse, DocumentOut
from regradar.database.models import Document
from regradar.database.repository import DocumentRepository

router = APIRouter(
    tags=["documents"],
    dependencies=[Depends(require_auth)],
)


def _to_out(document: Document) -> DocumentOut:
    """Map a Document ORM row to its API DTO.

    Agency names are stored as a ' | '-joined string (see ingestion); split
    them back into a list for the client.
    """
    agencies = [name for name in document.agency_names.split(" | ") if name]
    return DocumentOut(
        document_number=document.document_number,
        title=document.title,
        document_type=document.document_type,
        abstract=document.abstract,
        publication_date=document.publication_date,
        html_url=document.html_url,
        comments_close_on=document.comments_close_on,
        agency_names=agencies,
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(default=20, ge=1, le=50, description="Page size."),
    offset: int = Query(default=0, ge=0, description="Number of documents to skip."),
    repository: DocumentRepository = Depends(get_repository),
) -> DocumentListResponse:
    """List stored documents, most recently published first (paginated)."""
    count = await repository.count_documents()
    documents = await repository.list_documents(limit=limit, offset=offset)
    return DocumentListResponse(
        count=count,
        limit=limit,
        offset=offset,
        results=[_to_out(doc) for doc in documents],
    )


@router.get("/documents/{document_number}", response_model=DocumentOut)
async def get_document(
    document_number: str = Path(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9-]+$",
        description="FR document number, e.g. '2026-12345'.",
    ),
    repository: DocumentRepository = Depends(get_repository),
) -> DocumentOut:
    """Fetch a single document by its Federal Register number."""
    document = await repository.get_document(document_number)
    if document is None:
        raise APIError(
            status_code=404,
            code="not_found",
            message=f"No document with number '{document_number}'.",
        )
    return _to_out(document)
