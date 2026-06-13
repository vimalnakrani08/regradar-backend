"""Anthropic-backed implementation of the AnswerGenerator protocol."""

from __future__ import annotations

from anthropic import AsyncAnthropic

from regradar.config import get_settings
from regradar.generation.base import Source

# The system prompt enforces the critical anti-hallucination rule: this is a
# regulatory tool, so the model must ground every claim in the provided
# chunks, cite document numbers, and decline rather than guess.
_SYSTEM_PROMPT = """\
You answer questions about U.S. Federal Register regulatory documents.

Rules you must follow without exception:
- Use ONLY the information in the provided sources. Never rely on outside or
  prior knowledge about regulations.
- Cite the document number (e.g. "2026-12345") inline for every claim you make.
- If the sources do not contain enough information to answer, say
  "I don't have information on that." and stop. Do not speculate or fill gaps.
- Be concise and precise. Do not invent regulatory facts, dates, or agencies.\
"""


def _format_sources(sources: list[Source]) -> str:
    """Render sources into a labeled block for the user message.

    Each chunk is tagged with its document number and section so the model
    can cite it accurately.
    """
    blocks = []
    for source in sources:
        section = f" — {source.section}" if source.section else ""
        blocks.append(f"[Document {source.document_number}{section}]\n{source.content}")
    return "\n\n".join(blocks)


class AnthropicAnswerGenerator:
    """Generates cited answers using the Anthropic Messages API.

    Satisfies the AnswerGenerator protocol. The Anthropic client is injected
    (dependency injection) so business logic never constructs it inline and
    tests can supply a mock.
    """

    def __init__(self, client: AsyncAnthropic | None = None) -> None:
        settings = get_settings()
        self._client = client or AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = settings.answer_model
        self._max_tokens = settings.answer_max_tokens

    async def generate(self, question: str, sources: list[Source]) -> str:
        """Generate an answer grounded only in the given sources."""
        user_message = (
            f"Sources:\n\n{_format_sources(sources)}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the sources above, citing document numbers."
        )

        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        # response.content is a list of content blocks; pull the text blocks.
        return "".join(block.text for block in response.content if block.type == "text")
