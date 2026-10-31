from typing import Any

from mcp.client.client import Client

from app.container import Services
from app.mcp_server.server import create_mcp_server
from tests.integration.corpus import ONCALL_MD


async def test_tools_are_listed(services: Services) -> None:
    async with Client(create_mcp_server(services)) as client:
        tools = (await client.list_tools()).tools

    assert {tool.name for tool in tools} == {"search_documents", "ask"}


async def test_search_and_ask_over_mcp(services: Services) -> None:
    async with services.session_factory() as session:
        await services.ingestor.ingest(session, "oncall.md", ONCALL_MD)

    async with Client(create_mcp_server(services)) as client:
        search = await client.call_tool("search_documents", {"query": "SEV1 acknowledge"})
        ask = await client.call_tool("ask", {"question": "When do on-call shifts start?"})

    assert not search.is_error
    hits: list[dict[str, Any]] = search.structured_content["result"]
    assert hits[0]["filename"] == "oncall.md"

    assert not ask.is_error
    answer: dict[str, Any] = ask.structured_content
    assert "Monday at 10:00" in answer["answer"]
    assert answer["citations"][0]["filename"] == "oncall.md"
