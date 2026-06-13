"""FastAPI application factory.

`create_app()` wires routers, CORS, error handlers, and the process-wide
singletons (embedder + answer generator) created in the lifespan. Tests call
`create_app()` and override dependencies so no real DB/OpenAI/Anthropic is
touched; uvicorn serves the module-level `app`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from regradar.api.errors import register_exception_handlers
from regradar.api.routers import ask, documents, health, search
from regradar.config import get_settings
from regradar.database.session import get_engine
from regradar.embeddings.openai_embedder import OpenAIEmbedder
from regradar.generation.anthropic_generator import AnthropicAnswerGenerator


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create shared clients on startup; dispose the DB engine on shutdown.

    The embedder and generator wrap reusable HTTP clients, so they're built
    once and stored on app state rather than per request.
    """
    app.state.embedder = OpenAIEmbedder()
    app.state.generator = AnthropicAnswerGenerator()
    try:
        yield
    finally:
        await get_engine().dispose()


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Regradar API",
        description="Semantic search and cited answers over the U.S. Federal Register.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Allow the separate Next.js frontend to call the API from the browser.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(ask.router)
    app.include_router(search.router)
    app.include_router(documents.router)

    return app


# Module-level instance for `uvicorn regradar.api.app:app`.
app = create_app()
