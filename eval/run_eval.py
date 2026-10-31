"""Evaluate retrieval and answers over the checked-in corpus and Q&A set.

    python eval/run_eval.py                # plain /ask pipeline
    python eval/run_eval.py --agent        # LangGraph agent instead
    python eval/run_eval.py --json out.json

Uses whatever provider the environment configures (LLM_PROVIDER etc.). The corpus is
re-ingested on every run with the chunk settings given here, so results don't depend on
what else is in the database: retrieval is restricted to the corpus documents.
"""

import argparse
import asyncio
import hashlib
import json
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import delete

from app.container import Services, build_services
from app.core.config import Settings, get_settings
from app.db.models import Document
from app.evaluation.metrics import (
    EvalItem,
    EvalSummary,
    QuestionResult,
    contains_evidence,
    faithfulness,
    summarize,
)
from app.ingest.pipeline import Ingestor
from app.qa.prompts import NO_ANSWER
from app.qa.service import generate_answer
from app.schemas import AskResponse

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EVAL_DIR.parent
KS = (1, 3, 5)


def load_dataset(path: Path) -> list[EvalItem]:
    with path.open() as f:
        return [EvalItem(**json.loads(line)) for line in f if line.strip()]


def load_corpus(corpus: Path) -> list[tuple[str, bytes]]:
    return [(p.name, p.read_bytes()) for p in sorted(corpus.iterdir()) if p.is_file()]


async def ingest_corpus(
    services: Services, ingestor: Ingestor, files: list[tuple[str, bytes]]
) -> list[uuid.UUID]:
    document_ids = []
    async with services.session_factory() as session:
        for filename, data in files:
            # Drop any earlier copy so the chunk settings of this run actually apply.
            digest = hashlib.sha256(data).hexdigest()
            await session.execute(delete(Document).where(Document.sha256 == digest))
            await session.commit()
            result = await ingestor.ingest(session, filename, data)
            document_ids.append(result.document.id)
    return document_ids


async def evaluate_item(
    services: Services, item: EvalItem, document_ids: list[uuid.UUID], use_agent: bool
) -> QuestionResult:
    async with services.session_factory() as session:
        retriever = services.retriever(session)
        chunks = await retriever.search(item.question, max(KS), document_ids)
        relevance = [contains_evidence(item, c.filename, c.content) for c in chunks]

        response: AskResponse
        if use_agent:
            response = await services.agent(session, top_k=max(KS), document_ids=document_ids).run(
                item.question
            )
        else:
            response = await generate_answer(services.llm, item.question, chunks)

    cited = {c.chunk_id for c in response.citations}
    contexts = [c.content for c in chunks if c.chunk_id in cited] or [c.content for c in chunks]
    answered = response.answer.strip() != NO_ANSWER
    return QuestionResult(
        question=item.question,
        relevance=relevance,
        faithfulness=faithfulness(response.answer, contexts) if answered else None,
        answered=answered,
    )


def render_markdown(summary: EvalSummary, settings: Settings, use_agent: bool) -> str:
    provider = (
        "fake (hashing embedder + extractive answerer, not a real model)"
        if settings.llm_provider == "fake"
        else f"{settings.chat_model} / {settings.embedding_model} via {settings.llm_base_url}"
    )
    rows = [
        f"Pipeline: {'agent' if use_agent else 'ask'} | provider: {provider} | "
        f"questions: {summary.questions}",
        "",
        "| Metric | Value |",
        "|---|---|",
        *(f"| hit_rate@{k} | {summary.hit_rate[k]:.2f} |" for k in KS),
        f"| MRR | {summary.mrr:.2f} |",
        f"| faithfulness | {_fmt(summary.faithfulness)} |",
        f"| answered | {summary.answer_rate:.2f} |",
    ]
    return "\n".join(rows)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


async def run(
    settings: Settings,
    files: list[tuple[str, bytes]],
    items: list[EvalItem],
    *,
    chunk_size: int,
    chunk_overlap: int,
    use_agent: bool,
) -> list[QuestionResult]:
    services = build_services(settings)
    ingestor = Ingestor(
        embedder=services.embedder,
        count_tokens=services.ingestor.count_tokens,
        chunk_size_tokens=chunk_size,
        chunk_overlap_tokens=chunk_overlap,
    )
    try:
        document_ids = await ingest_corpus(services, ingestor, files)
        return [await evaluate_item(services, item, document_ids, use_agent) for item in items]
    finally:
        await services.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answer quality.")
    parser.add_argument("--dataset", type=Path, default=EVAL_DIR / "dataset.jsonl")
    parser.add_argument("--corpus", type=Path, default=EVAL_DIR / "corpus")
    # The corpus is small, so chunks are smaller than the service default to give
    # retrieval something to rank.
    parser.add_argument("--chunk-size", type=int, default=120)
    parser.add_argument("--chunk-overlap", type=int, default=20)
    parser.add_argument("--agent", action="store_true", help="evaluate the LangGraph agent")
    parser.add_argument("--json", type=Path, help="also write per-question results here")
    parser.add_argument("--skip-migrations", action="store_true")
    args = parser.parse_args()

    if not args.skip_migrations:
        command.upgrade(Config(str(PROJECT_DIR / "alembic.ini")), "head")

    settings = get_settings()
    results = asyncio.run(
        run(
            settings,
            load_corpus(args.corpus),
            load_dataset(args.dataset),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            use_agent=args.agent,
        )
    )
    summary = summarize(results, KS)
    print(render_markdown(summary, settings, args.agent))
    for result in results:
        if not any(result.relevance):
            print(f"  miss: {result.question}", file=sys.stderr)
    if args.json:
        payload = {"summary": asdict(summary), "results": [asdict(r) for r in results]}
        args.json.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
