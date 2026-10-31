import uuid
from collections.abc import Sequence

from app.providers.base import ChatModel
from app.qa.prompts import ANSWER_TOOL, NO_ANSWER, GroundedAnswer, answer_messages
from app.retrieval.citations import build_citations
from app.retrieval.types import RetrievedChunk, Retriever
from app.schemas import AskResponse


async def generate_answer(
    llm: ChatModel, question: str, chunks: Sequence[RetrievedChunk]
) -> AskResponse:
    if not chunks:
        return AskResponse(answer=NO_ANSWER, citations=[], retrieved=0)

    result = await llm.generate_structured(
        answer_messages(question, chunks),
        GroundedAnswer,
        tool_name=ANSWER_TOOL,
        tool_description="Submit the final answer with the numbers of the sources it relies on.",
    )
    return AskResponse(
        answer=result.answer,
        citations=build_citations(result.cited_sources, chunks, answer=result.answer),
        retrieved=len(chunks),
    )


async def answer_question(
    retriever: Retriever,
    llm: ChatModel,
    question: str,
    *,
    top_k: int,
    document_ids: Sequence[uuid.UUID] | None = None,
) -> AskResponse:
    """The plain RAG path: one retrieval, one model call."""
    chunks = await retriever.search(question, top_k, document_ids)
    return await generate_answer(llm, question, chunks)
