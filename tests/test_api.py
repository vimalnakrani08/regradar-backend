"""Tests for the HTTP API layer.

All collaborators are dependency-overridden, so these never touch the
database, OpenAI, or Anthropic. We assert the wire contract: response shapes,
validation, the not-found and upstream-failure paths, and — critically — that
unexpected errors return a generic envelope that leaks no internals.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date
from types import SimpleNamespace

import anthropic
import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from regradar.api.app import create_app
from regradar.api.dependencies import (
    get_db_session,
    get_rag_service,
    get_repository,
    get_search_service,
)
from regradar.generation.base import Answer, Source


async def _stub_session() -> AsyncIterator[object]:
    """A harmless default session: execute() is a no-op returning None."""
    yield SimpleNamespace(execute=lambda stmt: _async_return(None))


@pytest.fixture
def app() -> FastAPI:
    application = create_app()
    # httpx's ASGITransport doesn't run the lifespan, so populate the
    # singletons it would have created. They're sentinels here — every test
    # that exercises an endpoint overrides the service that would use them.
    application.state.embedder = object()
    application.state.generator = object()
    # Default the DB session to a no-op stub so tests never touch Postgres
    # unless they override it (the documents tests override the repository).
    application.dependency_overrides[get_db_session] = _stub_session
    return application


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    # raise_app_exceptions=False so the catch-all handler's 500 response is
    # returned to the client instead of being re-raised into the test.
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def _source(num: str = "2026-11854", sim: float = 0.65) -> Source:
    return Source(document_number=num, section="SUMMARY", similarity=sim, content="chunk text")


# --- /ask --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_returns_answer_and_sources(app: FastAPI, client: httpx.AsyncClient) -> None:
    answer = Answer(text="Grounded answer (2026-11854).", has_answer=True, sources=[_source()])
    app.dependency_overrides[get_rag_service] = lambda: SimpleNamespace(
        answer=lambda question: _async_return(answer)
    )

    resp = await client.post("/ask", json={"question": "What is the prediction markets rule?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["has_answer"] is True
    assert body["answer"] == "Grounded answer (2026-11854)."
    assert body["sources"][0]["document_number"] == "2026-11854"
    assert "excerpt" in body["sources"][0]


@pytest.mark.asyncio
async def test_ask_out_of_scope_passthrough(app: FastAPI, client: httpx.AsyncClient) -> None:
    answer = Answer(text="I don't have information on that.", has_answer=False, sources=[])
    app.dependency_overrides[get_rag_service] = lambda: SimpleNamespace(
        answer=lambda question: _async_return(answer)
    )

    resp = await client.post("/ask", json={"question": "crypto staking rules?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["has_answer"] is False
    assert body["sources"] == []


@pytest.mark.asyncio
async def test_ask_rejects_empty_question(client: httpx.AsyncClient) -> None:
    resp = await client.post("/ask", json={"question": ""})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


@pytest.mark.asyncio
async def test_ask_upstream_failure_is_503(app: FastAPI, client: httpx.AsyncClient) -> None:
    def _raise(question: str) -> Answer:
        raise anthropic.APIError("boom", httpx.Request("POST", "http://x"), body=None)

    app.dependency_overrides[get_rag_service] = lambda: SimpleNamespace(answer=_raise)

    resp = await client.post("/ask", json={"question": "a real question here"})

    assert resp.status_code == 503
    body = resp.json()
    assert body["error"]["code"] == "answer_unavailable"
    # No upstream detail leaked.
    assert "boom" not in resp.text


@pytest.mark.asyncio
async def test_unexpected_error_returns_generic_envelope(
    app: FastAPI, client: httpx.AsyncClient
) -> None:
    def _explode(question: str) -> Answer:
        raise RuntimeError("secret internal detail: db password leaked")

    app.dependency_overrides[get_rag_service] = lambda: SimpleNamespace(answer=_explode)

    resp = await client.post("/ask", json={"question": "a real question here"})

    assert resp.status_code == 500
    assert resp.json() == {
        "error": {"code": "internal_error", "message": "An unexpected error occurred."}
    }
    # The internal detail must not reach the client.
    assert "secret internal detail" not in resp.text


# --- /search -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_ranked_results(app: FastAPI, client: httpx.AsyncClient) -> None:
    results = [_source("2026-11854", 0.65), _source("2026-11843", 0.61)]
    app.dependency_overrides[get_search_service] = lambda: SimpleNamespace(
        search=lambda q, limit: _async_return(results)
    )

    resp = await client.get("/search", params={"q": "prediction markets", "limit": 5})

    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "prediction markets"
    assert [r["document_number"] for r in body["results"]] == ["2026-11854", "2026-11843"]


@pytest.mark.asyncio
async def test_search_rejects_bad_limit(client: httpx.AsyncClient) -> None:
    resp = await client.get("/search", params={"q": "markets", "limit": 999})
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "validation_error"


# --- /documents --------------------------------------------------------------


def _doc(num: str) -> SimpleNamespace:
    return SimpleNamespace(
        document_number=num,
        title="A Rule",
        document_type="Proposed Rule",
        abstract=None,
        publication_date=date(2026, 6, 2),
        html_url="https://example.com/doc",
        comments_close_on=None,
        agency_names="Transportation Department | PHMSA",
    )


@pytest.mark.asyncio
async def test_list_documents(app: FastAPI, client: httpx.AsyncClient) -> None:
    repo = SimpleNamespace(
        count_documents=lambda: _async_return(2),
        list_documents=lambda *, limit, offset: _async_return([_doc("2026-1"), _doc("2026-2")]),
    )
    app.dependency_overrides[get_repository] = lambda: repo

    resp = await client.get("/documents", params={"limit": 20, "offset": 0})

    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert body["results"][0]["agency_names"] == ["Transportation Department", "PHMSA"]


@pytest.mark.asyncio
async def test_get_document_not_found(app: FastAPI, client: httpx.AsyncClient) -> None:
    repo = SimpleNamespace(get_document=lambda num: _async_return(None))
    app.dependency_overrides[get_repository] = lambda: repo

    resp = await client.get("/documents/2026-9999")

    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "not_found"


@pytest.mark.asyncio
async def test_get_document_rejects_bad_number(client: httpx.AsyncClient) -> None:
    resp = await client.get("/documents/has spaces!")
    assert resp.status_code == 422


# --- /health -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_ok(app: FastAPI, client: httpx.AsyncClient) -> None:
    async def _ok_session() -> AsyncIterator[object]:
        yield SimpleNamespace(execute=lambda stmt: _async_return(None))

    app.dependency_overrides[get_db_session] = _ok_session

    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_health_db_down_is_503(app: FastAPI, client: httpx.AsyncClient) -> None:
    def _boom(stmt: object) -> object:
        raise RuntimeError("connection refused")

    async def _bad_session() -> AsyncIterator[object]:
        yield SimpleNamespace(execute=_boom)

    app.dependency_overrides[get_db_session] = _bad_session

    resp = await client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "database_unavailable"
    assert "connection refused" not in resp.text


async def _async_return(value: object) -> object:
    """Wrap a value in an awaitable, for stubbing async methods on SimpleNamespace."""
    return value
