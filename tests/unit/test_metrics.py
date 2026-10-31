import pytest

from app.evaluation.metrics import (
    EvalItem,
    QuestionResult,
    contains_evidence,
    faithfulness,
    hit_at_k,
    reciprocal_rank,
    summarize,
)

ITEM = EvalItem(question="q", source="a.md", evidence="Monday at  10:00")


def test_evidence_must_come_from_the_expected_source() -> None:
    assert contains_evidence(ITEM, "a.md", "Shifts start every\nmonday at 10:00.")
    assert not contains_evidence(ITEM, "b.md", "Shifts start every Monday at 10:00.")
    assert not contains_evidence(ITEM, "a.md", "Shifts start on Tuesday.")


def test_hit_at_k_and_reciprocal_rank() -> None:
    relevance = [False, False, True, False]

    assert [hit_at_k(relevance, k) for k in (1, 2, 3)] == [0.0, 0.0, 1.0]
    assert reciprocal_rank(relevance) == pytest.approx(1 / 3)
    assert reciprocal_rank([False, False]) == 0.0


def test_faithfulness_counts_supported_sentences() -> None:
    context = ["Shifts start every Monday at 10:00. Each shift has a primary and a secondary."]

    supported = "Shifts start every Monday at 10:00 [1]."
    half = "Shifts start every Monday at 10:00. Engineers are paid triple overtime."

    assert faithfulness(supported, context) == 1.0
    assert faithfulness(half, context) == 0.5


def test_faithfulness_is_undefined_for_empty_answers() -> None:
    assert faithfulness("", ["anything"]) is None


def test_summarize() -> None:
    results = [
        QuestionResult("q1", [True, False], faithfulness=1.0, answered=True),
        QuestionResult("q2", [False, True], faithfulness=0.5, answered=True),
        QuestionResult("q3", [False, False], faithfulness=None, answered=False),
    ]

    summary = summarize(results, ks=(1, 2))

    assert summary.questions == 3
    assert summary.hit_rate == {1: pytest.approx(1 / 3), 2: pytest.approx(2 / 3)}
    assert summary.mrr == pytest.approx((1 + 0.5 + 0) / 3)
    assert summary.faithfulness == pytest.approx(0.75)
    assert summary.answer_rate == pytest.approx(2 / 3)
