import pytest

from app.retrieval.fusion import reciprocal_rank_fusion


def test_single_ranking_keeps_its_order() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"]], k=60)

    assert [item for item, _ in fused] == ["a", "b", "c"]
    assert fused[0][1] == pytest.approx(1 / 61)


def test_item_found_by_both_retrievers_beats_items_found_by_one() -> None:
    vector = ["a", "b", "c"]
    text = ["d", "c", "e"]

    fused = dict(reciprocal_rank_fusion([vector, text], k=60))

    assert max(fused, key=lambda item: fused[item]) == "c"
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 62)


def test_items_at_the_same_rank_tie_and_keep_first_seen_order() -> None:
    fused = reciprocal_rank_fusion([["a"], ["b"]], k=60)

    assert [item for item, _ in fused] == ["a", "b"]
    assert fused[0][1] == fused[1][1] == pytest.approx(1 / 61)


def test_duplicates_within_one_ranking_count_once() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "a", "b"]], k=60))

    assert fused == {"a": pytest.approx(1 / 61), "b": pytest.approx(1 / 62)}


def test_k_trades_off_a_single_top_hit_against_agreement() -> None:
    # "top" is first in one list only; "agreed" is fourth in both.
    rankings = [["top", "x1", "x2", "agreed"], ["y1", "y2", "y3", "agreed"]]

    assert reciprocal_rank_fusion(rankings, k=1)[0][0] == "top"
    assert reciprocal_rank_fusion(rankings, k=60)[0][0] == "agreed"


def test_empty_input() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([["a"]], k=0)
