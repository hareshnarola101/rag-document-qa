from httpx import AsyncClient

from app.qa.prompts import NO_ANSWER
from tests.integration.corpus import CACHING_TXT, ONCALL_MD, VECTOR_PAGES
from tests.pdf_factory import make_pdf


async def upload_corpus(client: AsyncClient) -> dict[str, str]:
    ids = {}
    for name, data in [
        ("oncall.md", ONCALL_MD),
        ("caching.txt", CACHING_TXT),
        ("vectors.pdf", make_pdf(VECTOR_PAGES)),
    ]:
        response = await client.post("/documents", files={"file": (name, data)})
        ids[name] = response.json()["id"]
    return ids


async def test_search_returns_ranked_hits_from_both_retrievers(client: AsyncClient) -> None:
    await upload_corpus(client)

    response = await client.post("/search", json={"query": "304 Not Modified ETag", "top_k": 3})

    hits = response.json()
    assert response.status_code == 200
    assert hits[0]["filename"] == "caching.txt"
    assert hits[0]["vector_rank"] is not None
    assert hits[0]["text_rank"] is not None
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)


async def test_ask_answers_with_citations(client: AsyncClient) -> None:
    await upload_corpus(client)

    response = await client.post(
        "/ask", json={"question": "How fast must a SEV1 page be acknowledged?"}
    )

    body = response.json()
    assert response.status_code == 200
    assert "15 minutes" in body["answer"]
    assert body["citations"][0]["filename"] == "oncall.md"


async def test_ask_cites_the_pdf_page(client: AsyncClient) -> None:
    await upload_corpus(client)

    response = await client.post("/ask", json={"question": "What does HNSW build?"})

    [citation, *_] = response.json()["citations"]
    assert (citation["filename"], citation["page"]) == ("vectors.pdf", 2)


async def test_ask_can_be_scoped_to_documents(client: AsyncClient) -> None:
    ids = await upload_corpus(client)

    response = await client.post(
        "/ask",
        json={
            "question": "How fast must a SEV1 page be acknowledged?",
            "document_ids": [ids["caching.txt"]],
        },
    )

    body = response.json()
    assert body["answer"] == NO_ANSWER
    assert body["citations"] == []


async def test_ask_with_empty_index(client: AsyncClient) -> None:
    response = await client.post("/ask", json={"question": "Anything?"})

    assert response.json() == {"answer": NO_ANSWER, "citations": [], "retrieved": 0}


async def test_agent_endpoint_reports_its_steps(client: AsyncClient) -> None:
    await upload_corpus(client)

    response = await client.post("/agent/ask", json={"question": "When do on-call shifts start?"})

    body = response.json()
    assert response.status_code == 200
    assert "Monday at 10:00" in body["answer"]
    assert body["citations"][0]["filename"] == "oncall.md"
    assert body["steps"][0].startswith("retrieve")
    assert body["steps"][-1].startswith("answer")


async def test_request_validation(client: AsyncClient) -> None:
    assert (await client.post("/ask", json={"question": ""})).status_code == 422
    assert (await client.post("/ask", json={"question": "q", "top_k": 0})).status_code == 422
