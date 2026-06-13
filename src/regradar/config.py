"""Application configuration loaded from environment variables.

Secrets and environment-specific values never live in code. They come
from the environment (a local .env file in development, real env vars
in production).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Values are read from environment variables or a .env file. Field
    names map to env var names case-insensitively (database_url ->
    DATABASE_URL).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(
        default="postgresql+asyncpg://regradar:localdev@localhost:5432/regradar",
        description="Async SQLAlchemy connection string for PostgreSQL.",
    )

    embedding_dimension: int = Field(
        default=1536,
        description="Vector dimension for embeddings (OpenAI text-embedding-3-small).",
    )

    openai_api_key: str = Field(
        default="",
        description="OpenAI API key, used for embeddings.",
    )

    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name.",
    )

    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key, used for RAG answer generation.",
    )

    answer_model: str = Field(
        default="claude-opus-4-8",
        description="Anthropic model used to generate cited answers.",
    )

    answer_max_tokens: int = Field(
        default=1024,
        description="Maximum output tokens for a generated answer.",
    )

    similarity_threshold: float = Field(
        default=0.3,
        description=(
            "Minimum cosine similarity (1.0 = identical) for a chunk to be "
            "treated as relevant. Below this for every retrieved chunk, we "
            "skip the LLM and return an honest 'no information' answer. Floor "
            "starts ~0.3; tune on real data."
        ),
    )

    retrieval_top_k: int = Field(
        default=5,
        description="Number of chunks to retrieve as candidate context for an answer.",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    lru_cache ensures we read and validate the environment only once,
    then reuse the same Settings object everywhere.
    """
    return Settings()
