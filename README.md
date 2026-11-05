# rag-document-qa

[![CI](https://github.com/hareshnarola101/rag-document-qa/actions/workflows/ci.yml/badge.svg)](https://github.com/hareshnarola101/rag-document-qa/actions/workflows/ci.yml)
[![M8ven Score](https://m8ven.ai/badge/mcp/hareshnarola101-rag-document-qa-m4npdg)](https://m8ven.ai/mcp/hareshnarola101-rag-document-qa-m4npdg?s=readme)

A FastAPI service that answers questions about your own documents and shows where each answer
came from. Upload PDFs, Markdown or text files; ask a question; get an answer with citations
pointing at the document, page and passage it was based on.

## What it does and why

I built this to have a small, complete retrieval-augmented generation (RAG) service that I
can reason about end to end, without a framework hiding the interesting parts. It covers the
pieces I think matter in a real system:

- **Ingestion**: parse PDF/Markdown/text, split into token-sized chunks with overlap, embed,
  and store in PostgreSQL with pgvector. Re-uploading the same file is a no-op.
- **Hybrid retrieval**: vector similarity and Postgres full-text search, merged with
  reciprocal rank fusion (RRF).
- **Grounded answers**: the model must cite numbered sources, and the API returns those
  citations as structured data (document id, page, snippet), not just text.
- **An agent endpoint** built with LangGraph that grades what it retrieved, rewrites the query
  once if nothing useful came back, and says "I couldn't find it" instead of guessing.
- **An MCP server** so the same search and Q&A can be used as tools from MCP clients.
- **An evaluation script** with a small checked-in Q&A set, measuring retrieval hit rate and
  answer faithfulness.

Models sit behind a small provider interface that targets any OpenAI-compatible API, so the
same code runs against OpenAI, Ollama or vLLM. There is also a deterministic `fake` provider:
the test suite runs with no API key and no network, and `docker compose up` works out of the
box.

## Architecture

```mermaid
flowchart LR
    http[HTTP client] --> api[FastAPI]
    mcpc[MCP client] --> mcp[MCP server<br/>stdio or streamable HTTP]
    api --> services[Services]
    mcp --> services

    subgraph ingest[Ingestion]
        parse[parse<br/>PDF / MD / TXT] --> chunk[chunk<br/>token-aware, overlap] --> embed[embed]
    end
    subgraph answer[Question answering]
        retrieve[vector search +<br/>full-text search] --> rrf[reciprocal rank fusion] --> llm[LLM<br/>forced tool call] --> cite[citations]
    end

    services --> ingest
    services --> answer
    embed --> pg[(PostgreSQL<br/>pgvector + tsvector)]
    retrieve --> pg
    embed -.-> provider[OpenAI-compatible API<br/>OpenAI / Ollama / vLLM]
    llm -.-> provider
```

```
app/
  api/          FastAPI routers and dependencies
  agent/        LangGraph agent (retrieve, grade, rewrite, answer)
  core/         settings (pydantic-settings) and small text helpers
  db/           SQLAlchemy 2 async models and session factory
  ingest/       parsing, token counting, chunking, ingestion pipeline
  retrieval/    hybrid search, RRF, citation building
  qa/           prompts, structured output models, the plain RAG path
  providers/    Embedder / ChatModel interfaces, OpenAI-compatible and fake implementations
  mcp_server/   MCP server exposing search_documents and ask
  evaluation/   metrics used by eval/run_eval.py
  container.py  wires settings into the objects the API, MCP server and eval share
alembic/        migration (pgvector extension, HNSW and GIN indexes)
eval/           run_eval.py, dataset.jsonl and a small corpus
tests/          unit tests, and integration tests against real Postgres + pgvector
```

## Quickstart

You need Docker. This starts Postgres with pgvector and the API using the offline `fake`
models:

```bash
docker compose up --build
```

The API runs migrations on startup and listens on http://localhost:8000 (OpenAPI docs at
`/docs`).

To use real models, copy `.env.example` to `.env` and pick a provider.

**Ollama (local):**

```bash
LLM_PROVIDER=openai docker compose --profile ollama up -d --build
docker compose exec ollama ollama pull llama3.1:8b
docker compose exec ollama ollama pull nomic-embed-text
```

**OpenAI:** set `LLM_PROVIDER=openai`, `LLM_BASE_URL=https://api.openai.com/v1`,
`LLM_API_KEY`, `CHAT_MODEL=gpt-4o-mini`, `EMBEDDING_MODEL=text-embedding-3-small` and
`EMBEDDING_DIM=1536` in `.env`, then `docker compose up -d --build`. The embedding size is part
of the database schema, so start from an empty database (`docker compose down -v`) when you
change it.

vLLM, LM Studio and other OpenAI-compatible servers work the same way: point `LLM_BASE_URL` at
them.

**Without Docker** (Python 3.12 and [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
docker compose up -d db               # or any Postgres with pgvector
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
```

## API examples

The responses below are real output from the `fake` provider, trimmed a little. The fake
"model" answers by extracting the best-matching sentences, so treat the wording as a
placeholder for what a real LLM would write.

**Upload a document**

```bash
curl -F "file=@eval/corpus/oncall-handbook.md" http://localhost:8000/documents
```

```json
{
  "id": "c68831b2-037e-4e5a-bf88-bb12a157ec2b",
  "filename": "oncall-handbook.md",
  "content_type": "text/markdown",
  "page_count": null,
  "chunk_count": 2,
  "created_at": "2026-10-31T13:19:50.160271Z"
}
```

This returns `201` for a new document, and `200` with the existing record if the same bytes
were uploaded before. Unsupported types get `415`; files with no extractable text get `422`.
`GET /documents`, `GET /documents/{id}` and `DELETE /documents/{id}` do what you'd expect.

**Ask a question**

```bash
curl http://localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "How quickly must the primary acknowledge a SEV1 page?"}'
```

```json
{
  "answer": "The primary must acknowledge a SEV1 page within 15 minutes and open an incident channel immediately. ...",
  "citations": [
    {
      "source": 1,
      "chunk_id": "c0b72d78-f0a5-43cc-a64a-8a30a4460c18",
      "document_id": "c68831b2-037e-4e5a-bf88-bb12a157ec2b",
      "filename": "oncall-handbook.md",
      "page": null,
      "snippet": "…The primary must acknowledge a SEV1 page within 15 minutes and open an incident channel immediately. SEV2 means degraded service with a workaround available; ..."
    }
  ],
  "retrieved": 3
}
```

`page` is filled in for PDFs. The snippet starts at the sentence in the chunk that best
matches the answer. Both `/ask` and `/search` accept `top_k` and an optional `document_ids`
list to restrict the search to specific documents.

**Ask the agent**

```bash
curl http://localhost:8000/agent/ask -H 'content-type: application/json' \
  -d '{"question": "What is our refund policy for enterprise customers?"}'
```

```json
{
  "answer": "I couldn't find the answer in the provided documents.",
  "citations": [],
  "retrieved": 0,
  "query_used": "refund policy enterprise customer",
  "rewrites": 1,
  "steps": [
    "retrieve: 3 chunks",
    "grade: 0/3 relevant",
    "rewrite: 'refund policy enterprise customer'",
    "retrieve: 3 chunks",
    "grade: 0/3 relevant",
    "answer: 0 citations"
  ]
}
```

For a question the documents do cover ("Who gets paged for incidents involving payments?"),
the steps are `retrieve: 3 chunks`, `grade: 1/3 relevant`, `answer: 1 citations`, and the
answer cites the handbook.

**Search only (no model call)**

```bash
curl http://localhost:8000/search -H 'content-type: application/json' \
  -d '{"query": "SEV1 escalation", "top_k": 2}'
```

Each hit includes its fused `score` plus `vector_rank` and `text_rank`, so you can see which
retriever found what. I use this to debug retrieval before blaming the model.

## How retrieval works

Every query runs two searches over the `chunks` table and merges the results:

1. **Vector search**: cosine distance between the query embedding and chunk embeddings,
   served by an HNSW index (`vector_cosine_ops`).
2. **Full-text search**: a stored, computed `tsvector` column with a GIN index, ranked with
   `ts_rank_cd`. `plainto_tsquery` ANDs every term, which is too strict for a question like
   "what is the thing about the Vary header", so I rewrite its `&` operators to `|` and let
   the ranking reward chunks that match more terms.
3. **Reciprocal rank fusion**: each chunk scores `sum(1 / (60 + rank))` over the lists it
   appears in, and the top `top_k` go to the model.

Why both: embeddings are good at paraphrase ("how fast do I need to respond" vs "acknowledge
within 15 minutes") but weak on exact tokens such as error codes, config keys and names like
`SEV3` or `stale-if-error`. Full-text search is the opposite. Running both covers each one's
blind spots.

Why RRF rather than a weighted sum of scores: cosine distance and `ts_rank_cd` live on
unrelated scales, and `ts_rank_cd` isn't even comparable across queries. RRF only uses rank
positions, so there is nothing to normalise and only one parameter (`k`, default 60) to think
about. A chunk that both retrievers rank reasonably well usually beats one that only a single
retriever liked, which in my experience is the right bias.

Chunking splits text into sentences (keeping Markdown headings and list items on their own
lines and re-joining hard-wrapped lines), then packs whole sentences into chunks of up to
`CHUNK_SIZE_TOKENS` tokens, counted with tiktoken's `cl100k_base`. Each chunk starts with the
last sentences of the previous one, up to `CHUNK_OVERLAP_TOKENS`. Chunks never cross PDF pages,
so every citation maps to exactly one page.

## The agent

```mermaid
flowchart TD
    start([question]) --> retrieve
    retrieve --> grade
    grade -- "some chunks relevant" --> answer
    grade -- "none relevant, retry left" --> rewrite
    grade -- "none relevant, no retries left" --> answer
    rewrite --> retrieve
    answer --> done([answer + citations])
```

It's a LangGraph `StateGraph` with four nodes (`app/agent/graph.py`):

- **retrieve** runs the same hybrid search as `/ask`.
- **grade** asks the model which retrieved chunks actually help answer the question. Only
  those go to the answer step, which keeps the context small and on topic.
- **rewrite** runs when nothing relevant came back: the model turns the question into a
  keyword-style search query and the graph loops back to retrieve, at most once by default.
- **answer** writes the cited answer from the original question and the relevant chunks. If
  there are none, it returns a fixed "couldn't find" answer without calling the model.

Every model call in the project is a forced tool call whose parameters are a Pydantic model
(`GroundedAnswer`, `RelevanceGrade`, `RewrittenQuery`). I chose this over JSON mode or
`response_format` because tool calling is the structured-output mechanism that OpenAI, Ollama
and vLLM all support. If the arguments fail validation, the error is sent back to the model
once so it can correct itself.

## MCP server

`app/mcp_server` uses the official `mcp` Python SDK and exposes two tools:

- `search_documents(query, top_k=5)` returns matching passages with file name and page.
- `ask(question, top_k=5)` returns an answer with citations.

It uses the same services as the API (no HTTP hop), so start the stack first to get a migrated
database. To use it from an MCP client over stdio, point the client at the container:

```json
{
  "mcpServers": {
    "rag-document-qa": {
      "command": "docker",
      "args": [
        "compose", "-f", "/path/to/rag-document-qa/docker-compose.yml",
        "run", "--rm", "-T", "api", "python", "-m", "app.mcp_server"
      ]
    }
  }
}
```

Without Docker it's `uv run python -m app.mcp_server`. For clients that connect over HTTP:
`python -m app.mcp_server --transport streamable-http --host 0.0.0.0 --port 8001`.

## Evaluation

`eval/run_eval.py` ingests the four documents in `eval/corpus/` and runs the 16 questions in
`eval/dataset.jsonl`. Each question names its source file and an evidence phrase that a
correct chunk must contain. Retrieval is limited to the corpus documents, so whatever else is
in the database doesn't affect the numbers.

- **hit_rate@k**: share of questions where a top-k chunk contains the evidence.
- **MRR**: mean of 1 / rank of the first chunk with the evidence.
- **faithfulness**: share of answer sentences whose content words mostly (80% or more) appear
  in the cited chunks. It's a lexical check: it catches answers that drift from their sources
  but can't judge paraphrase.
- **answered**: share of questions where the system answered rather than saying it couldn't
  find the answer.

```bash
docker compose run --rm api python eval/run_eval.py           # plain /ask pipeline
docker compose run --rm api python eval/run_eval.py --agent   # LangGraph agent
```

Results from my run with the **fake provider** (hashing embedder and extractive answerer,
not a real model):

| Metric | `/ask`, fake models | agent, fake models |
|---|---|---|
| hit_rate@1 | 0.94 | 0.94 |
| hit_rate@3 | 1.00 | 1.00 |
| hit_rate@5 | 1.00 | 1.00 |
| MRR | 0.97 | 0.97 |
| faithfulness | 1.00 | 1.00 |
| answered | 1.00 | 0.94 |

Read these as a check that the pipeline is wired correctly, not as a quality claim. The fake
embedder is bag-of-words, the corpus is tiny, and an extractive answerer is faithful by
construction. The one agent miss is real behaviour, though: the fake grader is strict about
keyword overlap and rejected the right chunk for "What does the no-cache directive actually
mean?", so the agent declined to answer. To evaluate a real model, set the provider variables
(see Quickstart) and run the same commands.

## Running tests

Tests never call a real model: they use the fake embedder and chat model, and the
OpenAI-compatible client is tested against an in-process mock HTTP transport. Integration
tests run the real Alembic migration against Postgres with pgvector and go through the HTTP
API, the retriever and the MCP server (via an in-process MCP client).

With Docker (same Postgres image as CI):

```bash
docker compose --profile test run --rm test                  # pytest
docker compose --profile test run --rm test ruff check .
docker compose --profile test run --rm test mypy
```

With uv and a Postgres+pgvector reachable at `TEST_DATABASE_URL` (default
`postgresql+asyncpg://rag:rag@localhost:5432/rag_test`; `docker compose up -d db` creates it):

```bash
uv run pytest                          # everything
uv run pytest -m "not integration"     # unit tests only, no database needed
```

CI (`.github/workflows/ci.yml`) runs ruff, mypy in strict mode and the full test suite against
a pgvector service container, and builds the Docker image.

## Design decisions and trade-offs

- **Postgres for everything.** Vectors, full-text search and metadata live in one database, so
  deletes cascade, filters are plain SQL and there is one thing to back up. A dedicated vector
  database would scale further, but I'd want to hit Postgres's limits first.
- **The embedding dimension is part of the schema.** `vector(768)` lets pgvector build an HNSW
  index; the cost is that switching to a model with a different size means a fresh database
  and re-ingesting. The OpenAI-compatible embedder checks the returned size and fails with a
  clear message instead of letting inserts fail later.
- **Synchronous ingestion.** Upload parses, embeds and stores within the request. That's fine
  for documents of a few hundred pages and keeps the system simple; a background queue is the
  first thing I'd add for large files.
- **Sentence-level chunking with overlap.** Packing whole sentences gives cleaner embeddings
  and citations than fixed windows, at the cost of chunk sizes varying a bit. Token counts use
  `cl100k_base`, which isn't every embedding model's tokenizer but is close enough for sizing,
  and the default chunk size leaves headroom under common 512-token limits.
- **Forced tool calls for structured output.** More portable across providers than JSON mode.
  The cost is an extra round trip when a small model returns invalid arguments.
- **Deduplication by content hash.** The same bytes are stored once, and a re-upload returns
  the existing document. Different files with the same name are different documents.
- **The agent graph is built per request.** Compiling a four-node graph is cheap, and building
  it per request lets nodes close over the request's database session instead of passing it
  through graph config.
- **Fake models live in the main package.** They make the service usable with no key at all
  and keep tests fast and deterministic. Their output is labelled wherever it shows up,
  including the evaluation report.
- **Migrations run on container start.** Convenient for a single instance; with several
  replicas I'd run them as a separate release step.

## What I'd add next

- A background ingestion queue with status polling for large uploads.
- A cross-encoder reranker after RRF, and an evaluation run comparing vector-only, text-only,
  hybrid, and hybrid with reranking.
- An LLM-judge faithfulness metric (or Ragas) alongside the lexical one, and a larger Q&A set.
- Streaming answers over server-sent events.
- OCR for scanned PDFs, and DOCX/HTML parsers.
- Authentication and per-user document collections.

## License

MIT. See [LICENSE](LICENSE).
