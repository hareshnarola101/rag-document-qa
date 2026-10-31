from app.container import Services
from tests.integration.corpus import CACHING_TXT, ONCALL_MD


async def ingest_corpus(services: Services) -> None:
    async with services.session_factory() as session:
        await services.ingestor.ingest(session, "oncall.md", ONCALL_MD)
        await services.ingestor.ingest(session, "caching.txt", CACHING_TXT)


async def test_exact_term_match_is_found_by_full_text_search(services: Services) -> None:
    await ingest_corpus(services)

    async with services.session_factory() as session:
        hits = await services.retriever(session).search("SEV3", top_k=3)

    assert hits[0].text_rank == 1
    assert "SEV3" in hits[0].content


async def test_questions_full_of_stop_words_still_match(services: Services) -> None:
    # plainto_tsquery would AND every term; the OR rewrite still finds partial matches.
    await ingest_corpus(services)

    async with services.session_factory() as session:
        hits = await services.retriever(session).search(
            "what is the thing about a Vary header and also unicorns", top_k=2
        )

    assert any(hit.text_rank is not None and "Vary" in hit.content for hit in hits)


async def test_vector_search_alone_still_returns_results(services: Services) -> None:
    await ingest_corpus(services)

    async with services.session_factory() as session:
        hits = await services.retriever(session).search("zzzz qqqq", top_k=2)

    # Nothing matches lexically, but nearest-neighbour search always returns something.
    assert len(hits) == 2
    assert all(hit.text_rank is None and hit.vector_rank is not None for hit in hits)


async def test_chunks_are_deleted_with_their_document(services: Services) -> None:
    await ingest_corpus(services)
    async with services.session_factory() as session:
        result = await services.ingestor.ingest(session, "oncall.md", ONCALL_MD)
        await session.delete(result.document)
        await session.commit()

    async with services.session_factory() as session:
        hits = await services.retriever(session).search("on-call shifts Monday", top_k=10)

    assert {hit.filename for hit in hits} == {"caching.txt"}
