import uuid

from httpx import AsyncClient

from tests.integration.corpus import CACHING_TXT, ONCALL_MD, VECTOR_PAGES
from tests.pdf_factory import make_pdf


async def upload(client: AsyncClient, name: str, data: bytes) -> dict[str, object]:
    response = await client.post("/documents", files={"file": (name, data)})
    assert response.status_code == 201, response.text
    body: dict[str, object] = response.json()
    return body


async def test_health(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.json() == {"status": "ok"}


async def test_upload_markdown_creates_chunks(client: AsyncClient) -> None:
    document = await upload(client, "oncall.md", ONCALL_MD)

    assert document["filename"] == "oncall.md"
    assert document["content_type"] == "text/markdown"
    assert document["page_count"] is None
    assert isinstance(document["chunk_count"], int)
    assert document["chunk_count"] >= 2


async def test_upload_pdf_records_pages(client: AsyncClient) -> None:
    document = await upload(client, "vectors.pdf", make_pdf(VECTOR_PAGES))

    assert document["content_type"] == "application/pdf"
    assert document["page_count"] == 2


async def test_reupload_returns_the_existing_document(client: AsyncClient) -> None:
    first = await upload(client, "oncall.md", ONCALL_MD)

    again = await client.post("/documents", files={"file": ("renamed.md", ONCALL_MD)})

    assert again.status_code == 200
    assert again.json()["id"] == first["id"]
    assert len((await client.get("/documents")).json()) == 1


async def test_list_get_and_delete(client: AsyncClient) -> None:
    oncall = await upload(client, "oncall.md", ONCALL_MD)
    caching = await upload(client, "caching.txt", CACHING_TXT)

    listed = (await client.get("/documents")).json()
    assert {d["id"] for d in listed} == {oncall["id"], caching["id"]}
    assert (await client.get(f"/documents/{oncall['id']}")).json()["filename"] == "oncall.md"

    assert (await client.delete(f"/documents/{oncall['id']}")).status_code == 204
    assert (await client.get(f"/documents/{oncall['id']}")).status_code == 404
    assert (await client.delete(f"/documents/{oncall['id']}")).status_code == 404

    hits = (await client.post("/search", json={"query": "on-call shifts Monday"})).json()
    assert all(hit["filename"] == "caching.txt" for hit in hits)


async def test_unknown_document_is_404(client: AsyncClient) -> None:
    assert (await client.get(f"/documents/{uuid.uuid4()}")).status_code == 404


async def test_rejects_unsupported_empty_and_unreadable_files(client: AsyncClient) -> None:
    unsupported = await client.post("/documents", files={"file": ("a.docx", b"data")})
    empty = await client.post("/documents", files={"file": ("a.txt", b"")})
    blank = await client.post("/documents", files={"file": ("a.txt", b"  \n\n ")})

    assert unsupported.status_code == 415
    assert empty.status_code == 422
    assert blank.status_code == 422
    assert "no extractable text" in blank.json()["detail"]
