"""Prompts and the structured outputs we ask the model for.

Every model call is a forced tool call, so each output model below doubles as the JSON
schema of a tool's parameters.
"""

from collections.abc import Sequence

from pydantic import BaseModel, Field

from app.providers.base import ChatMessage
from app.retrieval.types import RetrievedChunk

ANSWER_TOOL = "submit_answer"
GRADE_TOOL = "grade_sources"
REWRITE_TOOL = "rewrite_query"

NO_ANSWER = "I couldn't find the answer in the provided documents."


class GroundedAnswer(BaseModel):
    answer: str = Field(description="The answer, citing sources inline as [n].")
    cited_sources: list[int] = Field(
        default_factory=list,
        description="Numbers of the sources that support the answer, e.g. [1, 3].",
    )


class RelevanceGrade(BaseModel):
    relevant_sources: list[int] = Field(
        default_factory=list,
        description="Numbers of the sources that help answer the question. Empty if none do.",
    )
    reasoning: str = Field(default="", description="One short sentence explaining the grade.")


class RewrittenQuery(BaseModel):
    query: str = Field(description="A search query likely to find passages that answer it.")


_ANSWER_SYSTEM = f"""You answer questions using only the numbered sources you are given.
- Every claim must be supported by a source. Cite sources inline like [1] or [2][3].
- List the numbers of all sources you used in cited_sources.
- If the sources do not contain the answer, set answer to exactly "{NO_ANSWER}" and leave
  cited_sources empty. Do not use outside knowledge.
- Be concise: a few sentences at most."""

_GRADE_SYSTEM = """You decide which numbered sources contain information that helps answer
the question. Be strict: a source that only shares keywords but does not help answer the
question is not relevant."""

_REWRITE_SYSTEM = """The search for this question returned nothing useful. Rewrite it as a
short keyword-style search query: keep the key entities and terms, add likely synonyms,
drop filler words."""


def format_sources(chunks: Sequence[RetrievedChunk]) -> str:
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        location = chunk.filename if chunk.page is None else f"{chunk.filename}, page {chunk.page}"
        blocks.append(f"[{number}] {location}\n{chunk.content}")
    return "\n\n".join(blocks)


def _user_prompt(question: str, chunks: Sequence[RetrievedChunk] | None = None) -> str:
    # Keep the question on one line; the fake model (and log readers) rely on it.
    prompt = f"Question: {' '.join(question.split())}"
    if chunks is not None:
        prompt += f"\n\nSources:\n{format_sources(chunks)}"
    return prompt


def answer_messages(question: str, chunks: Sequence[RetrievedChunk]) -> list[ChatMessage]:
    return [
        {"role": "system", "content": _ANSWER_SYSTEM},
        {"role": "user", "content": _user_prompt(question, chunks)},
    ]


def grade_messages(question: str, chunks: Sequence[RetrievedChunk]) -> list[ChatMessage]:
    return [
        {"role": "system", "content": _GRADE_SYSTEM},
        {"role": "user", "content": _user_prompt(question, chunks)},
    ]


def rewrite_messages(question: str) -> list[ChatMessage]:
    return [
        {"role": "system", "content": _REWRITE_SYSTEM},
        {"role": "user", "content": _user_prompt(question)},
    ]
