"""Deterministic stand-ins for the embedding and chat models.

They make the whole service runnable with no API key or network: the test suite uses them,
and `LLM_PROVIDER=fake` lets you try the API locally. They are deliberately simple and
lexical, so evaluation numbers produced with them say nothing about real model quality.
"""

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from app.core.text import content_words, split_sentences
from app.providers.base import ChatMessage, ProviderError
from app.qa.prompts import ANSWER_TOOL, GRADE_TOOL, NO_ANSWER, REWRITE_TOOL

_QUESTION_RE = re.compile(r"^Question:\s*(.+)$", re.MULTILINE)
_SOURCE_RE = re.compile(r"^\[(\d+)\][^\n]*\n(.*?)(?=^\[\d+\]|\Z)", re.MULTILINE | re.DOTALL)


class HashingEmbedder:
    """Bag-of-words feature hashing: texts that share words get similar vectors."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in content_words(text):
            digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            # A zero vector has no cosine distance to anything; give empty text a fixed direction.
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]


def _overlap(question_terms: set[str], text: str) -> int:
    return len(question_terms & set(content_words(text)))


class FakeChatModel:
    """Answers extractively by picking the source sentences that best match the question."""

    def __init__(self, relevance_threshold: float = 0.34) -> None:
        self._relevance_threshold = relevance_threshold

    async def generate_structured[T: BaseModel](
        self,
        messages: Sequence[ChatMessage],
        schema: type[T],
        *,
        tool_name: str,
        tool_description: str,
    ) -> T:
        prompt = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        question_match = _QUESTION_RE.search(prompt)
        if question_match is None:
            raise ProviderError("fake model expects a 'Question:' line in the prompt")
        question = question_match.group(1).strip()
        sources = {int(n): body.strip() for n, body in _SOURCE_RE.findall(prompt)}

        handlers = {
            ANSWER_TOOL: self._answer,
            GRADE_TOOL: self._grade,
            REWRITE_TOOL: self._rewrite,
        }
        if tool_name not in handlers:
            raise ProviderError(f"fake model has no handler for tool {tool_name!r}")
        return schema.model_validate(handlers[tool_name](question, sources))

    def _answer(self, question: str, sources: dict[int, str]) -> dict[str, Any]:
        terms = set(content_words(question))
        scored = [
            (_overlap(terms, sentence), number, sentence)
            for number, body in sources.items()
            for sentence in split_sentences(body)
        ]
        scored = [item for item in scored if item[0] > 0]
        if not scored:
            return {"answer": NO_ANSWER, "cited_sources": []}

        scored.sort(key=lambda item: item[0], reverse=True)
        # Overlapping chunks repeat sentences; keep the first (best-ranked) copy only.
        seen: set[str] = set()
        unique = []
        for item in scored:
            if item[2] not in seen:
                seen.add(item[2])
                unique.append(item)
        best_score = unique[0][0]
        picked = [item for item in unique[:2] if item[0] * 2 >= best_score]
        return {
            "answer": " ".join(sentence for _, _, sentence in picked),
            "cited_sources": list(dict.fromkeys(number for _, number, _ in picked)),
        }

    def _grade(self, question: str, sources: dict[int, str]) -> dict[str, Any]:
        terms = set(content_words(question))
        needed = max(1, math.ceil(len(terms) * self._relevance_threshold))
        relevant = [n for n, body in sources.items() if _overlap(terms, body) >= needed]
        return {"relevant_sources": relevant, "reasoning": f"lexical overlap >= {needed} terms"}

    def _rewrite(self, question: str, sources: dict[int, str]) -> dict[str, Any]:
        keywords = content_words(question)
        return {"query": " ".join(keywords) or question}
