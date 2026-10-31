from collections.abc import Hashable, Sequence


def reciprocal_rank_fusion[K: Hashable](
    rankings: Sequence[Sequence[K]], k: int = 60
) -> list[tuple[K, float]]:
    """Merge several ranked lists into one using reciprocal rank fusion.

    Each item scores sum(1 / (k + rank)) over the lists it appears in (rank is 1-based).
    Only positions matter, never raw scores, which is the point: cosine distances and
    ts_rank values live on unrelated scales and can't be compared directly. k=60 comes from
    the original paper (Cormack et al., 2009); larger k flattens the advantage of the top ranks.

    Ties keep the order in which items were first seen, so results are deterministic.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    scores: dict[K, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(dict.fromkeys(ranking), start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
