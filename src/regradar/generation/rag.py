"""RAG orchestration: embed -> retrieve -> threshold-gate -> generate.

`RagService` ties the pieces together but owns none of them — the embedder,
repository, and answer generator are all injected. This keeps the orchestration
logic free of construction concerns and trivially testable with stubs.
"""

from __future__ import annotations

from regradar.config import get_settings
from regradar.database.repository import DocumentRepository
from regradar.embeddings.base import Embedder
from regradar.generation.base import Answer, AnswerGenerator, Source

# Returned (without calling the LLM) when nothing relevant is retrieved.
_NO_ANSWER_TEXT = "I don't have information on that."


class RagService:
    """Answers questions over the indexed Federal Register corpus.

    The flow embeds the question, retrieves the most similar chunks, and only
    calls the LLM when at least one chunk clears the similarity threshold.
    Below the threshold we return an honest disclaimer and spend nothing on
    generation — the anti-hallucination guard.
    """

    def __init__(
        self,
        embedder: Embedder,
        repository: DocumentRepository,
        generator: AnswerGenerator,
        *,
        similarity_threshold: float | None = None,
        top_k: int | None = None,
    ) -> None:
        settings = get_settings()
        self._embedder = embedder
        self._repository = repository
        self._generator = generator
        self._threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.similarity_threshold
        )
        self._top_k = top_k if top_k is not None else settings.retrieval_top_k

    async def answer(self, question: str) -> Answer:
        """Answer a question, grounded only in retrieved chunks.

        Args:
            question: The user's natural-language question.

        Returns:
            An `Answer`. When retrieval finds nothing at or above the
            similarity threshold, `has_answer` is False, `text` is an honest
            disclaimer, `sources` is empty, and no LLM call is made.
        """
        query_vectors = await self._embedder.embed([question])
        query_embedding = query_vectors[0]

        matches = await self._repository.search_chunks(query_embedding, limit=self._top_k)

        # Keep only chunks relevant enough to ground an answer.
        sources = [
            Source(
                document_number=chunk.document_number,
                section=chunk.section,
                similarity=similarity,
                content=chunk.content,
            )
            for chunk, similarity in matches
            if similarity >= self._threshold
        ]

        if not sources:
            return Answer(text=_NO_ANSWER_TEXT, has_answer=False, sources=[])

        text = await self._generator.generate(question, sources)
        return Answer(text=text, has_answer=True, sources=sources)
