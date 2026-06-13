"""The /ask endpoint: retrieval-augmented, cited answers.

This is the expensive path — it can call the Anthropic API — so it carries
both the auth and rate-limit seams. Upstream LLM/embedding failures are
translated into a generic 503 rather than leaking provider errors.
"""

from __future__ import annotations

import anthropic
import openai
from fastapi import APIRouter, Depends

from regradar.api.dependencies import enforce_rate_limit, get_rag_service, require_auth
from regradar.api.errors import APIError
from regradar.api.schemas import AskRequest, AskResponse, SourceOut, excerpt
from regradar.generation.rag import RagService

router = APIRouter(
    tags=["ask"],
    dependencies=[Depends(require_auth), Depends(enforce_rate_limit)],
)


@router.post("/ask", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    service: RagService = Depends(get_rag_service),
) -> AskResponse:
    """Answer a question, grounded only in retrieved Federal Register chunks.

    When nothing relevant is retrieved, the service returns an honest
    disclaimer with `has_answer=False` and no LLM call is made.
    """
    try:
        answer = await service.answer(payload.question)
    except (anthropic.APIError, openai.APIError) as exc:
        # Don't leak the upstream provider's error shape/status to the client;
        # embedding (OpenAI) and generation (Anthropic) are both upstream here.
        raise APIError(
            status_code=503,
            code="answer_unavailable",
            message="The answer service is temporarily unavailable. Please try again.",
        ) from exc

    return AskResponse(
        answer=answer.text,
        has_answer=answer.has_answer,
        sources=[
            SourceOut(
                document_number=s.document_number,
                section=s.section,
                similarity=s.similarity,
                excerpt=excerpt(s.content),
            )
            for s in answer.sources
        ],
    )
