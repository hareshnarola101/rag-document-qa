"""Chat and embedding clients for any server that speaks the OpenAI HTTP API."""

import logging
from collections.abc import Sequence

import openai
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
)
from pydantic import BaseModel, ValidationError

from app.providers.base import ChatMessage, ProviderError

logger = logging.getLogger(__name__)


def _batched[T](items: Sequence[T], size: int) -> list[Sequence[T]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


class OpenAICompatibleEmbedder:
    def __init__(self, client: AsyncOpenAI, model: str, dim: int, batch_size: int = 64) -> None:
        self._client = client
        self._model = model
        self.dim = dim
        self._batch_size = batch_size

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in _batched(texts, self._batch_size):
            try:
                response = await self._client.embeddings.create(
                    model=self._model, input=list(batch)
                )
            except openai.OpenAIError as exc:
                raise ProviderError(f"embedding request failed: {exc}") from exc
            # The API may return items out of order; `index` is authoritative.
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))

        if vectors and len(vectors[0]) != self.dim:
            raise ProviderError(
                f"model {self._model!r} returned {len(vectors[0])}-dim vectors but "
                f"EMBEDDING_DIM is {self.dim}; set EMBEDDING_DIM to match the model"
            )
        return vectors


class OpenAICompatibleChatModel:
    def __init__(
        self, client: AsyncOpenAI, model: str, temperature: float = 0.0, max_attempts: int = 2
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_attempts = max_attempts

    async def generate_structured[T: BaseModel](
        self,
        messages: Sequence[ChatMessage],
        schema: type[T],
        *,
        tool_name: str,
        tool_description: str,
    ) -> T:
        tool: ChatCompletionFunctionToolParam = {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_description,
                "parameters": schema.model_json_schema(),
            },
        }
        conversation = [_to_openai_message(m) for m in messages]

        for attempt in range(1, self._max_attempts + 1):
            raw = await self._call(conversation, tool)
            try:
                return schema.model_validate_json(raw)
            except ValidationError as exc:
                logger.warning("structured output invalid (attempt %d): %s", attempt, exc)
                # Smaller local models sometimes miss a field; showing them the error usually
                # fixes it on the next try.
                conversation.append({"role": "assistant", "content": raw})
                conversation.append(
                    {
                        "role": "user",
                        "content": f"That did not match the schema: {exc}. "
                        f"Call {tool_name} again with valid arguments.",
                    }
                )
        raise ProviderError(f"model did not return valid {schema.__name__} output")

    async def _call(
        self, conversation: list[ChatCompletionMessageParam], tool: ChatCompletionFunctionToolParam
    ) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=conversation,
                tools=[tool],
                tool_choice={"type": "function", "function": {"name": tool["function"]["name"]}},
                temperature=self._temperature,
            )
        except openai.OpenAIError as exc:
            raise ProviderError(f"chat request failed: {exc}") from exc

        message = response.choices[0].message
        for call in message.tool_calls or []:
            if isinstance(call, ChatCompletionMessageFunctionToolCall):
                return call.function.arguments
        # Some servers ignore tool_choice and put the JSON in the message body instead.
        if message.content:
            return message.content
        raise ProviderError("model returned neither a tool call nor content")


def _to_openai_message(message: ChatMessage) -> ChatCompletionMessageParam:
    content = message["content"]
    match message["role"]:
        case "system":
            return {"role": "system", "content": content}
        case "assistant":
            return {"role": "assistant", "content": content}
        case "user":
            return {"role": "user", "content": content}
