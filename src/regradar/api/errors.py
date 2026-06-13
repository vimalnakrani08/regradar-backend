"""Public error envelope and exception handlers.

Every error the client sees has the same shape — `{"error": {"code", "message"}}`
— and never carries internal detail (stack traces, DB messages, upstream API
errors). Full diagnostics are logged server-side. This matters for a public
regulatory API: leaking internals is both a security and a trust problem.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger("regradar.api")


class APIError(HTTPException):
    """An HTTPException carrying a stable machine-readable error code.

    Routers raise this for expected failures (not found, upstream
    unavailable). The `code` is part of the client contract; `detail` is the
    human-readable message. Both are safe to expose by construction.
    """

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(status_code=status_code, detail=message)
        self.code = code


def _envelope(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers that render every error in the public envelope."""

    @app.exception_handler(APIError)
    async def _handle_api_error(_: Request, exc: APIError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=_envelope(exc.code, exc.detail))

    @app.exception_handler(HTTPException)
    async def _handle_http_exception(_: Request, exc: HTTPException) -> JSONResponse:
        # Plain HTTPExceptions (e.g. from FastAPI internals) — map status to a
        # generic code and pass through the already-safe detail string.
        code = "http_error" if exc.status_code < 500 else "internal_error"
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return JSONResponse(status_code=exc.status_code, content=_envelope(code, message))

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Field-level validation info is safe to return (it describes the
        # client's own input) and helps the frontend show useful messages.
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed.",
                    "details": jsonable_encoder(exc.errors()),
                }
            },
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Last line of defense: log everything, tell the client nothing.
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_envelope("internal_error", "An unexpected error occurred."),
        )
