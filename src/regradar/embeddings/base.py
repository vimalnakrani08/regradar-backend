"""The Embedder protocol: a swappable interface for embedding providers.

Defining embedding as a Protocol means the rest of the application
depends on the *interface*, not a specific provider. Swapping OpenAI for
a local model (BGE-M3 via Ollama) or a premium one (Voyage) later means
writing a new class that satisfies this protocol — no other code changes.
"""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    """Anything that can turn texts into embedding vectors.

    Implementations must produce vectors of a consistent dimension
    matching the database schema (1536 for text-embedding-3-small).
    """

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: The strings to embed.

        Returns:
            One embedding vector per input text, in the same order.
        """
        ...
