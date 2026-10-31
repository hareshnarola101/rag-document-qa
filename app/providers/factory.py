from openai import AsyncOpenAI

from app.core.config import Settings
from app.providers.base import ChatModel, Embedder
from app.providers.fake import FakeChatModel, HashingEmbedder
from app.providers.openai_compat import OpenAICompatibleChatModel, OpenAICompatibleEmbedder


def _client(settings: Settings) -> AsyncOpenAI:
    return AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        timeout=settings.llm_timeout_seconds,
    )


def build_embedder(settings: Settings) -> Embedder:
    if settings.llm_provider == "fake":
        return HashingEmbedder(settings.embedding_dim)
    return OpenAICompatibleEmbedder(
        _client(settings),
        model=settings.embedding_model,
        dim=settings.embedding_dim,
        batch_size=settings.embedding_batch_size,
    )


def build_chat_model(settings: Settings) -> ChatModel:
    if settings.llm_provider == "fake":
        return FakeChatModel()
    return OpenAICompatibleChatModel(
        _client(settings), model=settings.chat_model, temperature=settings.llm_temperature
    )
