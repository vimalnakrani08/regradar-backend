"""Typed results and the swappable answer-generation interface.

`AnswerGenerator` is a Protocol for the same reason `Embedder` is: the rest
of the application depends on the *interface*, not on Anthropic specifically.
Swapping in a different LLM provider later means writing one new class that
satisfies this protocol — no other code changes.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, Field


class Source(BaseModel):
    """A retrieved chunk used as grounding for an answer.

    Serves two roles: it is the context handed to the LLM, and it is the
    citation returned to the caller. `content` is the full chunk text (what
    the model grounds on); a client can truncate it for display.
    """

    document_number: str = Field(..., description="FR document number, e.g. '2026-12345'.")
    section: str | None = Field(default=None, description="Structural section, e.g. 'SUMMARY'.")
    similarity: float = Field(..., description="Cosine similarity to the query (1.0 = identical).")
    content: str = Field(..., description="The chunk's text.")


class Answer(BaseModel):
    """A generated answer plus the sources it was grounded in.

    `has_answer` is False when retrieval found nothing relevant enough to
    answer from. In that case `text` is an honest disclaimer, `sources` is
    empty, and no LLM call was made.
    """

    text: str = Field(..., description="The answer, or an honest 'no information' disclaimer.")
    has_answer: bool = Field(
        ..., description="True if the answer is grounded in retrieved sources."
    )
    sources: list[Source] = Field(
        default_factory=list, description="The chunks the answer was grounded in, best match first."
    )


class AnswerGenerator(Protocol):
    """Anything that can turn a question + grounding sources into answer text.

    Implementations must ground their output ONLY in the provided sources,
    cite document numbers, and never invent regulatory facts. The orchestrator
    guarantees `sources` is non-empty before calling this.
    """

    async def generate(self, question: str, sources: list[Source]) -> str:
        """Generate an answer grounded only in the given sources.

        Args:
            question: The user's question.
            sources: Retrieved chunks to ground the answer in (non-empty).

        Returns:
            The generated answer text, citing document numbers.
        """
        ...
