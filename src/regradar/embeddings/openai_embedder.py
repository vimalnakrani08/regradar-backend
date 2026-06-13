"""OpenAI-backed implementation of the Embedder protocol."""

from __future__ import annotations

from openai import AsyncOpenAI

from regradar.config import get_settings

# OpenAI accepts many inputs per request. Batching keeps us efficient
# and well under the per-request token limit.
_MAX_BATCH_SIZE = 100


class OpenAIEmbedder:
    """Generates embeddings using OpenAI's embedding API.

    Satisfies the Embedder protocol. Batches large inputs to minimize
    the number of API calls.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts, batching requests of up to _MAX_BATCH_SIZE."""
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for start in range(0, len(texts), _MAX_BATCH_SIZE):
            batch = texts[start : start + _MAX_BATCH_SIZE]
            response = await self._client.embeddings.create(
                model=self._model,
                input=batch,
            )
            # The API returns embeddings in the same order as inputs.
            all_vectors.extend(item.embedding for item in response.data)

        return all_vectors
