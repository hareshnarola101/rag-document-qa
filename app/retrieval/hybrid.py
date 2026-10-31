import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import ColumnElement, Row, Select, Text, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.providers.base import Embedder
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.types import RetrievedChunk

# (chunk id, document id, filename, page, content)
ChunkRow = Row[uuid.UUID, uuid.UUID, str, int | None, str]
ChunkSelect = Select[uuid.UUID, uuid.UUID, str, int | None, str]


def _or_tsquery(query: str) -> ColumnElement[Any]:
    # plainto_tsquery handles parsing, stemming and stop words but ANDs every term, which is
    # too strict for natural-language questions. Swapping & for | gives OR semantics, and
    # ts_rank_cd still rewards chunks that match more terms. The lexemes are already stemmed,
    # so the second parse uses the 'simple' config to avoid stemming them again.
    plain = cast(func.plainto_tsquery("english", query), Text)
    return func.to_tsquery("simple", func.replace(plain, " & ", " | "))


class HybridRetriever:
    """Vector similarity and full-text search, merged with reciprocal rank fusion."""

    def __init__(
        self,
        session: AsyncSession,
        embedder: Embedder,
        *,
        candidates: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self._session = session
        self._embedder = embedder
        self._candidates = candidates
        self._rrf_k = rrf_k

    async def search(
        self,
        query: str,
        top_k: int,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[RetrievedChunk]:
        [embedding] = await self._embedder.embed([query])
        vector_rows = await self._vector_search(embedding, document_ids)
        text_rows = await self._text_search(query, document_ids)

        rows = {row.id: row for row in (*vector_rows, *text_rows)}
        vector_ranks = {row.id: rank for rank, row in enumerate(vector_rows, start=1)}
        text_ranks = {row.id: rank for rank, row in enumerate(text_rows, start=1)}
        fused = reciprocal_rank_fusion(
            [[row.id for row in vector_rows], [row.id for row in text_rows]], k=self._rrf_k
        )

        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=rows[chunk_id].document_id,
                filename=rows[chunk_id].filename,
                page=rows[chunk_id].page,
                content=rows[chunk_id].content,
                score=score,
                vector_rank=vector_ranks.get(chunk_id),
                text_rank=text_ranks.get(chunk_id),
            )
            for chunk_id, score in fused[:top_k]
        ]

    async def _vector_search(
        self, embedding: list[float], document_ids: Sequence[uuid.UUID] | None
    ) -> Sequence[ChunkRow]:
        stmt = self._base_query(document_ids).order_by(Chunk.embedding.cosine_distance(embedding))
        result = await self._session.execute(stmt.limit(self._candidates))
        return result.all()

    async def _text_search(
        self, query: str, document_ids: Sequence[uuid.UUID] | None
    ) -> Sequence[ChunkRow]:
        tsquery = _or_tsquery(query)
        stmt = (
            self._base_query(document_ids)
            .where(Chunk.search_vector.op("@@")(tsquery))
            .order_by(func.ts_rank_cd(Chunk.search_vector, tsquery).desc())
        )
        result = await self._session.execute(stmt.limit(self._candidates))
        return result.all()

    @staticmethod
    def _base_query(document_ids: Sequence[uuid.UUID] | None) -> ChunkSelect:
        stmt = select(
            Chunk.id, Chunk.document_id, Document.filename, Chunk.page, Chunk.content
        ).join(Document, Document.id == Chunk.document_id)
        if document_ids:
            stmt = stmt.where(Chunk.document_id.in_(document_ids))
        return stmt
