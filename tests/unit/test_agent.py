import uuid
from collections.abc import Sequence

from app.agent.graph import RagAgent
from app.providers.fake import FakeChatModel
from app.qa.prompts import NO_ANSWER
from app.retrieval.types import RetrievedChunk
from tests.factories import make_chunk

RELEVANT = make_chunk("Shifts start every Monday at 10:00 local time.", filename="oncall.md")
NOISE = make_chunk("ETags let a server answer 304 Not Modified.", filename="http.md")


class ScriptedRetriever:
    """Returns a scripted result per call and records the queries it was given."""

    def __init__(self, *results: list[RetrievedChunk]) -> None:
        self._results = list(results)
        self.queries: list[str] = []

    async def search(
        self, query: str, top_k: int, document_ids: Sequence[uuid.UUID] | None = None
    ) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self._results.pop(0) if self._results else []


async def test_answers_directly_when_retrieval_is_relevant() -> None:
    retriever = ScriptedRetriever([NOISE, RELEVANT])
    agent = RagAgent(retriever, FakeChatModel(), top_k=4)

    response = await agent.run("When do on-call shifts start?")

    assert response.answer == "Shifts start every Monday at 10:00 local time."
    assert [c.filename for c in response.citations] == ["oncall.md"]
    assert response.retrieved == 1  # the grader filtered out the noise
    assert response.rewrites == 0
    assert response.steps == ["retrieve: 2 chunks", "grade: 1/2 relevant", "answer: 1 citations"]


async def test_rewrites_the_query_and_retries_once() -> None:
    retriever = ScriptedRetriever([NOISE], [RELEVANT])
    agent = RagAgent(retriever, FakeChatModel(), top_k=4)

    response = await agent.run("When do on-call shifts start?")

    assert retriever.queries == ["When do on-call shifts start?", "on-call shift start"]
    assert response.query_used == "on-call shift start"
    assert response.rewrites == 1
    assert response.citations[0].chunk_id == RELEVANT.chunk_id
    assert [s.split(":")[0] for s in response.steps] == [
        "retrieve",
        "grade",
        "rewrite",
        "retrieve",
        "grade",
        "answer",
    ]


async def test_gives_up_honestly_after_the_retry_budget() -> None:
    retriever = ScriptedRetriever([NOISE], [NOISE])
    agent = RagAgent(retriever, FakeChatModel(), top_k=4, max_rewrites=1)

    response = await agent.run("When do on-call shifts start?")

    assert response.answer == NO_ANSWER
    assert response.citations == []
    assert len(retriever.queries) == 2


async def test_empty_index_skips_the_grader() -> None:
    agent = RagAgent(ScriptedRetriever(), FakeChatModel(), top_k=4, max_rewrites=0)

    response = await agent.run("Anything at all?")

    assert response.answer == NO_ANSWER
    assert response.steps == [
        "retrieve: 0 chunks",
        "grade: nothing to grade",
        "answer: 0 citations",
    ]
