"""HTTP API layer.

A FastAPI application exposing search, ask (RAG), and document endpoints,
designed to be consumed by the separate Next.js frontend. The app is built
by `create_app()`; `app` is a ready-to-serve instance for uvicorn.
"""

from __future__ import annotations

from regradar.api.app import app, create_app

__all__ = ["app", "create_app"]
