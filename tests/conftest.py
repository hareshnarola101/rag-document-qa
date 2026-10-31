import os

import pytest

from app.core.config import Settings

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://rag:rag@localhost:5432/rag_test"
)


@pytest.fixture
def settings() -> Settings:
    # Explicit values so a developer's .env can't change test behaviour.
    return Settings(
        _env_file=None,
        database_url=TEST_DATABASE_URL,
        llm_provider="fake",
        tokenizer="regex",
        embedding_dim=256,
        chunk_size_tokens=60,
        chunk_overlap_tokens=12,
        retrieval_candidates=10,
        default_top_k=4,
    )
