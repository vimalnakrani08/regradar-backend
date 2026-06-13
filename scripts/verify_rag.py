"""Scratch script: exercise RagService end-to-end against the live database.

Not part of the test suite — a manual smoke check that the generation layer
grounds answers in real retrieved chunks, cites matching document numbers,
and honestly declines on out-of-scope questions.

Run: uv run python scripts/verify_rag.py
Requires: docker DB up with ingested data, OPENAI_API_KEY and
ANTHROPIC_API_KEY set. Pass --retrieval-only to skip the Anthropic call and
just inspect what retrieval returns (no key needed).
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from regradar.config import get_settings
from regradar.database.repository import DocumentRepository
from regradar.embeddings.openai_embedder import OpenAIEmbedder
from regradar.generation.anthropic_generator import AnthropicAnswerGenerator
from regradar.generation.rag import RagService

QUESTIONS = [
    "What are the public comment deadlines for recent rules?",
    "Tell me about the prediction markets proposed rule.",
    # Deliberately out of scope — should NOT be answerable from the corpus.
    "What are the new rules about cryptocurrency staking?",
]


async def _retrieval_only(embedder: OpenAIEmbedder, repo: DocumentRepository) -> None:
    """Show raw retrieval + scores for each question, without the LLM."""
    threshold = get_settings().similarity_threshold
    top_k = get_settings().retrieval_top_k
    for question in QUESTIONS:
        vec = (await embedder.embed([question]))[0]
        matches = await repo.search_chunks(vec, limit=top_k)
        print(f"\n{'=' * 70}\nQ: {question}")
        if not matches:
            print("  (no chunks returned)")
            continue
        best = matches[0][1]
        gate = "PASS -> would call LLM" if best >= threshold else "GATED -> no LLM call"
        print(f"  best similarity={best:.3f}  threshold={threshold}  [{gate}]")
        for chunk, score in matches:
            mark = "*" if score >= threshold else " "
            section = chunk.section or "-"
            print(f"   {mark} {score:.3f}  {chunk.document_number}  [{section}]")


async def _full(service: RagService) -> None:
    """Run the full RAG flow and print cited answers + sources."""
    for question in QUESTIONS:
        answer = await service.answer(question)
        print(f"\n{'=' * 70}\nQ: {question}")
        print(f"has_answer={answer.has_answer}\n")
        print(answer.text)
        if answer.sources:
            print("\nSources:")
            for s in answer.sources:
                print(f"  - {s.document_number}  [{s.section or '-'}]  sim={s.similarity:.3f}")
        else:
            print("\nSources: (none)")


async def main(retrieval_only: bool) -> None:
    engine = create_async_engine(get_settings().database_url)
    embedder = OpenAIEmbedder()
    async with AsyncSession(engine) as session:
        repo = DocumentRepository(session)
        if retrieval_only:
            await _retrieval_only(embedder, repo)
        else:
            service = RagService(embedder, repo, AnthropicAnswerGenerator())
            await _full(service)
    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.retrieval_only))
