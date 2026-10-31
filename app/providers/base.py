from collections.abc import Sequence
from typing import Literal, Protocol, TypedDict

from pydantic import BaseModel


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class ProviderError(RuntimeError):
    """The model backend failed or returned something we could not use."""


class Embedder(Protocol):
    dim: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class ChatModel(Protocol):
    async def generate_structured[T: BaseModel](
        self,
        messages: Sequence[ChatMessage],
        schema: type[T],
        *,
        tool_name: str,
        tool_description: str,
    ) -> T:
        """Return the model's reply parsed into `schema`.

        Implementations get a typed result by forcing a single tool call whose parameters are
        the schema. That works across OpenAI, Ollama and vLLM, which is not true of every
        provider's JSON-mode or `response_format` support.
        """
        ...
