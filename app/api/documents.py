import uuid

from fastapi import APIRouter, HTTPException, Query, Response, UploadFile, status
from sqlalchemy import select

from app.api.deps import ServicesDep, SessionDep
from app.db.models import Document
from app.schemas import DocumentOut

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentOut,
    status_code=status.HTTP_201_CREATED,
    responses={200: {"description": "Same file was already uploaded; existing document returned"}},
)
async def upload_document(
    file: UploadFile, response: Response, services: ServicesDep, session: SessionDep
) -> Document:
    max_bytes = services.settings.max_upload_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"file is larger than {services.settings.max_upload_mb} MB",
        )
    if not data:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "file is empty")

    result = await services.ingestor.ingest(session, file.filename or "upload.txt", data)
    if not result.created:
        response.status_code = status.HTTP_200_OK
    return result.document


@router.get("", response_model=list[DocumentOut])
async def list_documents(
    session: SessionDep,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[Document]:
    stmt = select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
    return list(await session.scalars(stmt))


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(document_id: uuid.UUID, session: SessionDep) -> Document:
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    return document


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: uuid.UUID, session: SessionDep) -> None:
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    await session.delete(document)  # chunks go with it via ON DELETE CASCADE
    await session.commit()
