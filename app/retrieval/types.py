import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page: int | None
    content: str
    score: float  # fused RRF score; only meaningful relative to other results of the same query
    vector_rank: int | None = None  # 1-based position in each retriever, None if it missed
    text_rank: int | None = None


class Retriever(Protocol):
    async def search(
        self, query: str, top_k: int, document_ids: Sequence[uuid.UUID] | None = None
    ) -> list[RetrievedChunk]: ...
