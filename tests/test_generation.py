"""Tests for the RAG generation layer.

The Anthropic call is always mocked, so these run fast, offline, and
deterministically — never hitting the live LLM API. RagService is tested with
stub embedder/repository/generator objects; the Anthropic generator is tested
with a mock AsyncAnthropic client.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from regradar.generation.anthropic_generator import AnthropicAnswerGenerator
from regradar.generation.base import Source
from regradar.generation.rag import RagService


class StubEmbedder:
    """Embedder that returns a fixed vector, recording what it was asked."""

    def __init__(self) -> None:
        self.embedded: list[str] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]


def _chunk(document_number: str, content: str, section: str | None = "SUMMARY") -> SimpleNamespace:
    """Build a stand-in for a DocumentChunk with the fields RagService reads."""
    return SimpleNamespace(document_number=document_number, section=section, content=content)


class StubRepository:
    """Repository whose search_chunks returns a preset list of (chunk, score)."""

    def __init__(self, matches: list[tuple[SimpleNamespace, float]]) -> None:
        self._matches = matches

    async def search_chunks(
        self, query_embedding: list[float], *, limit: int = 10
    ) -> list[tuple[SimpleNamespace, float]]:
        return self._matches[:limit]


@pytest.mark.asyncio
async def test_answer_above_threshold_calls_generator() -> None:
    """When a chunk clears the threshold, generate runs and sources are returned."""
    matches = [
        (_chunk("2026-10969", "PHMSA proposes new inspection requirements."), 0.71),
        (_chunk("2026-10968", "Unrelated postal rule.", section=None), 0.62),
    ]
    generator = AsyncMock()
    generator.generate.return_value = "PHMSA proposes new requirements (2026-10969)."

    service = RagService(
        StubEmbedder(),
        StubRepository(matches),  # type: ignore[arg-type]
        generator,
        similarity_threshold=0.3,
        top_k=5,
    )

    answer = await service.answer("What is PHMSA proposing?")

    assert answer.has_answer is True
    assert answer.text == "PHMSA proposes new requirements (2026-10969)."
    assert [s.document_number for s in answer.sources] == ["2026-10969", "2026-10968"]
    # The generator received the same sources we cite.
    generator.generate.assert_awaited_once()
    _, passed_sources = generator.generate.await_args.args
    assert [s.document_number for s in passed_sources] == ["2026-10969", "2026-10968"]


@pytest.mark.asyncio
async def test_answer_below_threshold_skips_llm() -> None:
    """When no chunk clears the threshold, return a disclaimer without calling the LLM."""
    matches = [(_chunk("2026-10969", "Barely related text."), 0.21)]
    generator = AsyncMock()

    service = RagService(
        StubEmbedder(),
        StubRepository(matches),  # type: ignore[arg-type]
        generator,
        similarity_threshold=0.3,
    )

    answer = await service.answer("Something unrelated?")

    assert answer.has_answer is False
    assert answer.text == "I don't have information on that."
    assert answer.sources == []
    generator.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_answer_no_results_skips_llm() -> None:
    """Empty retrieval also short-circuits to the honest disclaimer."""
    generator = AsyncMock()

    service = RagService(StubEmbedder(), StubRepository([]), generator)  # type: ignore[arg-type]

    answer = await service.answer("Anything?")

    assert answer.has_answer is False
    assert answer.sources == []
    generator.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_anthropic_generator_builds_grounded_prompt() -> None:
    """The generator sends model/system/sources to Anthropic and returns its text."""
    fake_response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Grounded answer (2026-10969).")]
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=fake_response)))

    generator = AnthropicAnswerGenerator(client=client)  # type: ignore[arg-type]
    sources = [
        Source(
            document_number="2026-10969",
            section="SUMMARY",
            similarity=0.71,
            content="PHMSA proposes new inspection requirements.",
        )
    ]

    text = await generator.generate("What is PHMSA proposing?", sources)

    assert text == "Grounded answer (2026-10969)."
    client.messages.create.assert_awaited_once()
    kwargs = client.messages.create.await_args.kwargs
    assert kwargs["model"] == "claude-opus-4-8"
    # Anti-hallucination instruction reaches the model.
    assert "ONLY the information in the provided sources" in kwargs["system"]
    # The chunk text and its document number reach the prompt.
    user_content = kwargs["messages"][0]["content"]
    assert "2026-10969" in user_content
    assert "PHMSA proposes new inspection requirements." in user_content
    assert "What is PHMSA proposing?" in user_content


@pytest.mark.asyncio
async def test_anthropic_generator_concatenates_text_blocks() -> None:
    """Multiple text blocks are joined; non-text blocks are ignored."""
    fake_response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Part one. "),
            SimpleNamespace(type="thinking", thinking="ignored"),
            SimpleNamespace(type="text", text="Part two."),
        ]
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(return_value=fake_response)))

    generator = AnthropicAnswerGenerator(client=client)  # type: ignore[arg-type]
    text = await generator.generate(
        "Q?",
        [Source(document_number="2026-1", similarity=0.9, content="ctx")],
    )

    assert text == "Part one. Part two."
