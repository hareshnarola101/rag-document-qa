from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables (and `.env` if present)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5432/rag"

    # "openai" means any OpenAI-compatible endpoint: OpenAI itself, Ollama, vLLM, LM Studio...
    # "fake" uses deterministic local models so the service runs with no API key at all.
    llm_provider: Literal["openai", "fake"] = "fake"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: SecretStr = SecretStr("not-needed")
    chat_model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"
    # Must match the model's output size. It is baked into the pgvector column by the
    # migration, so changing it later means re-running migrations on a fresh database.
    embedding_dim: int = Field(default=768, gt=0)
    embedding_batch_size: int = Field(default=64, gt=0)
    llm_timeout_seconds: float = 60.0
    llm_temperature: float = 0.0

    tokenizer: Literal["tiktoken", "regex"] = "tiktoken"
    chunk_size_tokens: int = Field(default=350, gt=0)
    chunk_overlap_tokens: int = Field(default=50, ge=0)

    retrieval_candidates: int = Field(default=20, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    default_top_k: int = Field(default=5, gt=0)

    max_upload_mb: int = Field(default=20, gt=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> "Settings":
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
