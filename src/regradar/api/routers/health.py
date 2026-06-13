"""Health check endpoint for load balancers and platform probes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from regradar.api.dependencies import get_db_session
from regradar.api.errors import APIError

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(session: AsyncSession = Depends(get_db_session)) -> dict[str, str]:
    """Report liveness, including a cheap database round-trip.

    Railway (and any uptime probe) hits this; a failing DB connection should
    surface as an unhealthy status, not a generic 500.
    """
    try:
        await session.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 — translate any DB failure into 503
        raise APIError(
            status_code=503,
            code="database_unavailable",
            message="The database is not reachable.",
        ) from exc
    return {"status": "ok"}
