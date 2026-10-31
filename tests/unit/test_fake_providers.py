import math

import pytest

from app.providers.base import ProviderError
from app.providers.fake import FakeChatModel, HashingEmbedder
from app.qa.prompts import (
    ANSWER_TOOL,
    GRADE_TOOL,
    NO_ANSWER,
    REWRITE_TOOL,
    GroundedAnswer,
    RelevanceGrade,
    RewrittenQuery,
    answer_messages,
    grade_messages,
    rewrite_messages,
)
from tests.factories import make_chunk


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


async def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dim=64)

    [first] = await embedder.embed(["postgres full-text search"])
    [again] = await embedder.embed(["postgres full-text search"])

    assert first == again
    assert len(first) == 64
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0)


async def test_hashing_embedder_puts_shared_words_closer() -> None:
    embedder = HashingEmbedder(dim=256)

    query, related, unrelated = await embedder.embed(
        ["how do vector indexes work", "vector indexes like HNSW", "the on-call rotation policy"]
    )

    assert cosine(query, related) > cosine(query, unrelated)


async def test_hashing_embedder_gives_empty_text_a_unit_vector() -> None:
    [vector] = await HashingEmbedder(dim=8).embed(["the and of"])

    assert vector == [1.0] + [0.0] * 7


ON_CALL = make_chunk("Shifts start every Monday at 10:00. Pages go to the primary first.")
CACHING = make_chunk("ETags let a server answer 304 Not Modified.")


async def test_fake_answer_extracts_and_cites_the_matching_source() -> None:
    llm = FakeChatModel()

    result = await llm.generate_structured(
        answer_messages("When do on-call shifts start?", [CACHING, ON_CALL]),
        GroundedAnswer,
        tool_name=ANSWER_TOOL,
        tool_description="",
    )

    assert result.answer == "Shifts start every Monday at 10:00."
    assert result.cited_sources == [2]


async def test_fake_answer_does_not_repeat_sentences_from_overlapping_chunks() -> None:
    overlap = make_chunk("Pages go to the primary first. Shifts start every Monday at 10:00.")

    result = await FakeChatModel().generate_structured(
        answer_messages("When do shifts start?", [ON_CALL, overlap]),
        GroundedAnswer,
        tool_name=ANSWER_TOOL,
        tool_description="",
    )

    assert result.answer == "Shifts start every Monday at 10:00."
    assert result.cited_sources == [1]


async def test_fake_answer_declines_when_nothing_matches() -> None:
    result = await FakeChatModel().generate_structured(
        answer_messages("What is the capital of France?", [CACHING]),
        GroundedAnswer,
        tool_name=ANSWER_TOOL,
        tool_description="",
    )

    assert result == GroundedAnswer(answer=NO_ANSWER, cited_sources=[])


async def test_fake_grader_keeps_only_overlapping_sources() -> None:
    grade = await FakeChatModel().generate_structured(
        grade_messages(
            "Which status code does a server send when not modified?", [ON_CALL, CACHING]
        ),
        RelevanceGrade,
        tool_name=GRADE_TOOL,
        tool_description="",
    )

    assert grade.relevant_sources == [2]


async def test_fake_rewrite_keeps_keywords() -> None:
    rewritten = await FakeChatModel().generate_structured(
        rewrite_messages("What is the policy for weekend pages?"),
        RewrittenQuery,
        tool_name=REWRITE_TOOL,
        tool_description="",
    )

    assert rewritten.query == "policy weekend page"


async def test_fake_rejects_unknown_tools() -> None:
    with pytest.raises(ProviderError, match="no handler"):
        await FakeChatModel().generate_structured(
            rewrite_messages("anything"), RewrittenQuery, tool_name="other", tool_description=""
        )
