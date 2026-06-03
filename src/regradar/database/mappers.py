"""Mapping functions between API models and database models.

The Federal Register API client returns Pydantic models. The database
layer uses SQLAlchemy models. Keeping these separate means a change to
the external API shape doesn't ripple into our storage schema. These
mappers translate between the two worlds.
"""

from __future__ import annotations

from regradar.database.models import Document
from regradar.models import FederalRegisterDocument

# Separator used to join multiple agency names into the single
# agency_names text column.
_AGENCY_SEPARATOR = " | "


def api_document_to_db(doc: FederalRegisterDocument) -> Document:
    """Convert an API document into a database Document model.

    Args:
        doc: The validated Pydantic model from the Federal Register API.

    Returns:
        An unattached SQLAlchemy Document ready to be added to a session.
    """
    return Document(
        document_number=doc.document_number,
        title=doc.title,
        document_type=doc.document_type,
        abstract=doc.abstract,
        publication_date=doc.publication_date,
        html_url=str(doc.html_url),
        comments_close_on=doc.comments_close_on,
        agency_names=_AGENCY_SEPARATOR.join(doc.agency_names),
    )
