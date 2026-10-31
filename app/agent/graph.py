"""A small corrective-RAG agent built with LangGraph.

    retrieve -> grade --(something relevant)--> answer
                  \\--(nothing relevant, retries left)--> rewrite -> retrieve

The grader filters out retrieved chunks that don't help, so the answer step only sees useful
context. If nothing survives, the model rewrites the question into a better search query and
we try once more before giving up honestly instead of answering from noise.
"""

import operator
import uuid
from collections.abc import Sequence
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.providers.base import ChatModel
from app.qa.prompts import (
    GRADE_TOOL,
    REWRITE_TOOL,
    RelevanceGrade,
    RewrittenQuery,
    grade_messages,
    rewrite_messages,
)
from app.qa.service import generate_answer
from app.retrieval.types import RetrievedChunk, Retriever
from app.schemas import AgentResponse, AskResponse


class AgentState(TypedDict):
    question: str
    query: str
    rewrites: int
    chunks: list[RetrievedChunk]
    relevant: list[RetrievedChunk]
    response: AskResponse | None
    steps: Annotated[list[str], operator.add]


class RagAgent:
    def __init__(
        self,
        retriever: Retriever,
        llm: ChatModel,
        *,
        top_k: int,
        max_rewrites: int = 1,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._top_k = top_k
        self._max_rewrites = max_rewrites
        self._document_ids = document_ids
        # Compiling is cheap, and building per request lets nodes close over a request-scoped
        # DB session instead of threading it through the graph config.
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade", self._grade)
        graph.add_node("rewrite", self._rewrite)
        graph.add_node("answer", self._answer)

        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "grade")
        graph.add_conditional_edges(
            "grade", self._route_after_grading, {"answer": "answer", "rewrite": "rewrite"}
        )
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("answer", END)
        return graph.compile()

    async def run(self, question: str) -> AgentResponse:
        initial: AgentState = {
            "question": question,
            "query": question,
            "rewrites": 0,
            "chunks": [],
            "relevant": [],
            "response": None,
            "steps": [],
        }
        final: AgentState = await self.graph.ainvoke(initial)
        response = final["response"]
        if response is None:
            raise RuntimeError("agent graph finished without reaching the answer node")
        return AgentResponse(
            **response.model_dump(),
            query_used=final["query"],
            rewrites=final["rewrites"],
            steps=final["steps"],
        )

    async def _retrieve(self, state: AgentState) -> dict[str, Any]:
        chunks = await self._retriever.search(state["query"], self._top_k, self._document_ids)
        return {"chunks": chunks, "steps": [f"retrieve: {len(chunks)} chunks"]}

    async def _grade(self, state: AgentState) -> dict[str, Any]:
        chunks = state["chunks"]
        if not chunks:
            return {"relevant": [], "steps": ["grade: nothing to grade"]}

        grade = await self._llm.generate_structured(
            grade_messages(state["question"], chunks),
            RelevanceGrade,
            tool_name=GRADE_TOOL,
            tool_description="Report which sources are relevant to the question.",
        )
        keep = {n for n in grade.relevant_sources if 1 <= n <= len(chunks)}
        relevant = [chunk for n, chunk in enumerate(chunks, start=1) if n in keep]
        return {"relevant": relevant, "steps": [f"grade: {len(relevant)}/{len(chunks)} relevant"]}

    def _route_after_grading(self, state: AgentState) -> Literal["answer", "rewrite"]:
        if state["relevant"] or state["rewrites"] >= self._max_rewrites:
            return "answer"
        return "rewrite"

    async def _rewrite(self, state: AgentState) -> dict[str, Any]:
        rewritten = await self._llm.generate_structured(
            rewrite_messages(state["question"]),
            RewrittenQuery,
            tool_name=REWRITE_TOOL,
            tool_description="Submit an improved search query.",
        )
        return {
            "query": rewritten.query,
            "rewrites": state["rewrites"] + 1,
            "steps": [f"rewrite: {rewritten.query!r}"],
        }

    async def _answer(self, state: AgentState) -> dict[str, Any]:
        # Answer the original question; the rewritten query was only for search.
        response = await generate_answer(self._llm, state["question"], state["relevant"])
        return {"response": response, "steps": [f"answer: {len(response.citations)} citations"]}
