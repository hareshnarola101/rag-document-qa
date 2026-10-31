"""Wires settings into the objects the API, MCP server and eval script share."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.agent.graph import RagAgent
from app.core.config import Settings
from app.db.session import create_engine, create_session_factory
from app.ingest.pipeline import Ingestor
from app.ingest.tokenizers import get_token_counter
from app.providers.base import ChatModel, Embedder
from app.providers.factory import build_chat_model, build_embedder
from app.retrieval.hybrid import HybridRetriever


@dataclass(frozen=True, slots=True)
class Services:
    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    embedder: Embedder
    llm: ChatModel
    ingestor: Ingestor

    def retriever(self, session: AsyncSession) -> HybridRetriever:
        return HybridRetriever(
            session,
            self.embedder,
            candidates=self.settings.retrieval_candidates,
            rrf_k=self.settings.rrf_k,
        )

    def agent(
        self,
        session: AsyncSession,
        *,
        top_k: int | None = None,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> RagAgent:
        return RagAgent(
            self.retriever(session),
            self.llm,
            top_k=top_k or self.settings.default_top_k,
            document_ids=document_ids,
        )

    async def aclose(self) -> None:
        await self.engine.dispose()


def build_services(
    settings: Settings, *, embedder: Embedder | None = None, llm: ChatModel | None = None
) -> Services:
    embedder = embedder or build_embedder(settings)
    engine = create_engine(settings.database_url)
    return Services(
        settings=settings,
        engine=engine,
        session_factory=create_session_factory(engine),
        embedder=embedder,
        llm=llm or build_chat_model(settings),
        ingestor=Ingestor(
            embedder=embedder,
            count_tokens=get_token_counter(settings.tokenizer),
            chunk_size_tokens=settings.chunk_size_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
        ),
    )
