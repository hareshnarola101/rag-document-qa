"""Exercises the real OpenAI client against an in-process mock transport (no network)."""

import json
from collections.abc import Callable
from typing import Any

import httpx2  # the HTTP client the openai SDK is built on
import pytest
from openai import AsyncOpenAI

from app.providers.base import ProviderError
from app.providers.openai_compat import OpenAICompatibleChatModel, OpenAICompatibleEmbedder
from app.qa.prompts import GroundedAnswer

Handler = Callable[[dict[str, Any]], dict[str, Any]]


def client_for(handler: Handler, requests: list[dict[str, Any]]) -> AsyncOpenAI:
    def respond(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        requests.append(body)
        return httpx2.Response(200, json=handler(body))

    return AsyncOpenAI(
        base_url="http://llm.test/v1",
        api_key="test",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(respond)),
    )


def chat_reply(*, arguments: str | None = None, content: str | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if arguments is not None:
        message["tool_calls"] = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "submit_answer", "arguments": arguments},
            }
        ]
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
    }


async def ask(model: OpenAICompatibleChatModel) -> GroundedAnswer:
    return await model.generate_structured(
        [{"role": "user", "content": "Question: q"}],
        GroundedAnswer,
        tool_name="submit_answer",
        tool_description="Submit the answer.",
    )


async def test_structured_output_is_requested_as_a_forced_tool_call() -> None:
    requests: list[dict[str, Any]] = []
    reply = chat_reply(arguments='{"answer": "42 [1]", "cited_sources": [1]}')
    model = OpenAICompatibleChatModel(client_for(lambda _: reply, requests), model="m")

    result = await ask(model)

    assert result == GroundedAnswer(answer="42 [1]", cited_sources=[1])
    [request] = requests
    assert request["tool_choice"] == {"type": "function", "function": {"name": "submit_answer"}}
    assert request["tools"][0]["function"]["parameters"] == GroundedAnswer.model_json_schema()


async def test_invalid_output_is_retried_with_the_validation_error() -> None:
    requests: list[dict[str, Any]] = []
    replies = iter(
        [
            chat_reply(arguments='{"cited_sources": [1]}'),  # missing "answer"
            chat_reply(arguments='{"answer": "fixed", "cited_sources": []}'),
        ]
    )
    model = OpenAICompatibleChatModel(client_for(lambda _: next(replies), requests), model="m")

    result = await ask(model)

    assert result.answer == "fixed"
    assert len(requests) == 2
    assert "did not match the schema" in requests[1]["messages"][-1]["content"]


async def test_gives_up_after_max_attempts() -> None:
    reply = chat_reply(arguments="not json")
    model = OpenAICompatibleChatModel(client_for(lambda _: reply, []), model="m", max_attempts=2)

    with pytest.raises(ProviderError, match="GroundedAnswer"):
        await ask(model)


async def test_falls_back_to_message_content_when_tool_choice_is_ignored() -> None:
    reply = chat_reply(content='{"answer": "from content", "cited_sources": [2]}')
    model = OpenAICompatibleChatModel(client_for(lambda _: reply, []), model="m")

    assert (await ask(model)).cited_sources == [2]


USAGE = {"prompt_tokens": 0, "total_tokens": 0}


def embedding_reply(body: dict[str, Any]) -> dict[str, Any]:
    # Return items in reverse order to check that `index` is respected.
    data = [
        {"object": "embedding", "index": i, "embedding": [float(i), 0.0, 0.0]}
        for i in range(len(body["input"]))
    ]
    return {"object": "list", "model": "e", "data": data[::-1], "usage": USAGE}


async def test_embedder_batches_and_keeps_input_order() -> None:
    requests: list[dict[str, Any]] = []
    embedder = OpenAICompatibleEmbedder(
        client_for(embedding_reply, requests), model="e", dim=3, batch_size=2
    )

    vectors = await embedder.embed(["a", "b", "c"])

    assert [v[0] for v in vectors] == [0.0, 1.0, 0.0]
    assert [r["input"] for r in requests] == [["a", "b"], ["c"]]


async def test_embedder_rejects_wrong_dimension() -> None:
    embedder = OpenAICompatibleEmbedder(client_for(embedding_reply, []), model="e", dim=768)

    with pytest.raises(ProviderError, match="EMBEDDING_DIM"):
        await embedder.embed(["a"])
