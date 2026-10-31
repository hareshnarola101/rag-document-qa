import uuid

from app.retrieval.types import RetrievedChunk


def make_chunk(content: str, filename: str = "notes.md", page: int | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        filename=filename,
        page=page,
        content=content,
        score=0.0,
    )
