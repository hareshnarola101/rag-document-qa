"""Request and response models for the HTTP API and the MCP tools."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    content_type: str
    page_count: int | None
    chunk_count: int
    created_at: datetime


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    document_ids: list[uuid.UUID] | None = None


class SearchHit(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page: int | None
    content: str
    score: float
    vector_rank: int | None
    text_rank: int | None


class Citation(BaseModel):
    source: int = Field(description="The [n] marker the answer used for this source")
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page: int | None
    snippet: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_ids: list[uuid.UUID] | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved: int = Field(description="How many chunks were given to the model")


class AgentResponse(AskResponse):
    query_used: str = Field(description="The search query that produced the final context")
    rewrites: int
    steps: list[str] = Field(description="Nodes visited in the agent graph, in order")
