"""RAG answer generation: turn a question + retrieved chunks into a cited answer.

This package layers cited answer generation on top of semantic search. The
public surface is:

  - `Answer` / `Source`: typed results (see `base`).
  - `AnswerGenerator`: a swappable Protocol for the LLM backend (see `base`),
    mirroring how `Embedder` abstracts the embedding provider.
  - `AnthropicAnswerGenerator`: the Anthropic-backed implementation.
  - `RagService`: orchestrates embed -> retrieve -> threshold-gate -> generate.
"""

from __future__ import annotations

from regradar.generation.anthropic_generator import AnthropicAnswerGenerator
from regradar.generation.base import Answer, AnswerGenerator, Source
from regradar.generation.rag import RagService

__all__ = [
    "Answer",
    "AnswerGenerator",
    "AnthropicAnswerGenerator",
    "RagService",
    "Source",
]
