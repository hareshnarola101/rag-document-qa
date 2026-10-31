"""MCP server exposing document search and question answering as tools.

It talks to the database directly through the same services as the HTTP API, so an MCP
client gets identical results without a network hop to the API.
"""

import argparse
import logging
from typing import Annotated

from mcp.server import MCPServer
from pydantic import Field

from app.container import Services, build_services
from app.core.config import get_settings
from app.qa.service import answer_question
from app.schemas import AskResponse, SearchHit

TopK = Annotated[int, Field(ge=1, le=20, description="How many chunks to retrieve")]


def create_mcp_server(services: Services) -> MCPServer:
    server = MCPServer(
        name="rag-document-qa",
        version="0.1.0",
        instructions=(
            "Search and question-answering over documents uploaded to rag-document-qa. "
            "Use search_documents to find passages and ask for a cited answer."
        ),
    )

    @server.tool()
    async def search_documents(query: str, top_k: TopK = 5) -> list[SearchHit]:
        """Find passages relevant to a query, with file name and page, via hybrid search."""
        async with services.session_factory() as session:
            chunks = await services.retriever(session).search(query, top_k)
        return [SearchHit.model_validate(chunk, from_attributes=True) for chunk in chunks]

    @server.tool()
    async def ask(question: str, top_k: TopK = 5) -> AskResponse:
        """Answer a question using only the uploaded documents, with citations to sources."""
        async with services.session_factory() as session:
            return await answer_question(
                services.retriever(session), services.llm, question, top_k=top_k
            )

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the rag-document-qa MCP server.")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    # stdout belongs to the MCP protocol on stdio, so logs go to stderr (basicConfig's default).
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    server = create_mcp_server(build_services(get_settings()))
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port)
