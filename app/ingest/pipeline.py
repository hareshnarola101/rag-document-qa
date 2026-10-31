import hashlib
import logging
from dataclasses import dataclass
from pathlib import PurePath

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document
from app.ingest.chunking import chunk_pages
from app.ingest.parsers import content_type_for, parse_document
from app.ingest.tokenizers import TokenCounter
from app.providers.base import Embedder

logger = logging.getLogger(__name__)


class EmptyDocumentError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class IngestResult:
    document: Document
    created: bool


@dataclass(frozen=True, slots=True)
class Ingestor:
    embedder: Embedder
    count_tokens: TokenCounter
    chunk_size_tokens: int
    chunk_overlap_tokens: int

    async def ingest(self, session: AsyncSession, filename: str, data: bytes) -> IngestResult:
        """Parse, chunk, embed and store a document. Re-uploading the same bytes is a no-op."""
        filename = PurePath(filename).name[:255]
        content_type = content_type_for(filename)
        digest = hashlib.sha256(data).hexdigest()

        existing = await _find_by_digest(session, digest)
        if existing is not None:
            return IngestResult(existing, created=False)

        pages = parse_document(filename, data)
        chunks = chunk_pages(
            pages,
            count_tokens=self.count_tokens,
            max_tokens=self.chunk_size_tokens,
            overlap_tokens=self.chunk_overlap_tokens,
        )
        if not chunks:
            raise EmptyDocumentError(
                "no extractable text found (scanned PDFs need OCR before upload)"
            )

        embeddings = await self.embedder.embed([chunk.text for chunk in chunks])
        page_numbers = [p.number for p in pages if p.number is not None]
        document = Document(
            filename=filename,
            content_type=content_type,
            sha256=digest,
            page_count=max(page_numbers) if page_numbers else None,
            chunk_count=len(chunks),
            chunks=[
                Chunk(
                    ordinal=chunk.ordinal,
                    page=chunk.page,
                    content=chunk.text,
                    token_count=chunk.token_count,
                    embedding=embedding,
                )
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ],
        )
        session.add(document)
        try:
            await session.commit()
        except IntegrityError:
            # Two uploads of the same file raced; the other one won.
            await session.rollback()
            existing = await _find_by_digest(session, digest)
            if existing is None:
                raise
            return IngestResult(existing, created=False)

        logger.info("ingested %s: %d chunks", filename, len(chunks))
        return IngestResult(document, created=True)


async def _find_by_digest(session: AsyncSession, digest: str) -> Document | None:
    result: Document | None = await session.scalar(
        select(Document).where(Document.sha256 == digest)
    )
    return result
