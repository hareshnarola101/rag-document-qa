"""Token-aware chunking.

Text is split into sentences (and lines, so Markdown lists and headings stay intact), then
sentences are packed greedily into chunks of at most `max_tokens`. Each new chunk starts with
the trailing sentences of the previous one, up to `overlap_tokens`, so a fact that straddles
a boundary is still retrievable in one piece. Overlap is sentence-granular on purpose:
cutting mid-sentence makes worse embeddings and worse citations.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

from app.core.text import sentence_spans
from app.ingest.parsers import ParsedPage
from app.ingest.tokenizers import TokenCounter


class Segment(NamedTuple):
    text: str
    tokens: int
    separator: str  # joins this segment to the previous one: " ", "\n" or "\n\n"


@dataclass(frozen=True, slots=True)
class TextChunk:
    ordinal: int
    page: int | None
    text: str
    token_count: int


def chunk_text(
    text: str, *, count_tokens: TokenCounter, max_tokens: int, overlap_tokens: int
) -> list[str]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError("overlap_tokens must be in [0, max_tokens)")

    segments = [
        Segment(part, count_tokens(part), separator if i == 0 else " ")
        for sentence, separator in sentence_spans(text)
        for i, part in enumerate(_fit_to_limit(sentence, count_tokens, max_tokens))
    ]

    chunks: list[str] = []
    window: list[Segment] = []
    for segment in segments:
        if window and _tokens(window) + segment.tokens > max_tokens:
            chunks.append(_join(window))
            window = _overlap_tail(window, overlap_tokens)
            while window and _tokens(window) + segment.tokens > max_tokens:
                window.pop(0)
        window.append(segment)
    if window:
        chunks.append(_join(window))
    return chunks


def chunk_pages(
    pages: Iterable[ParsedPage],
    *,
    count_tokens: TokenCounter,
    max_tokens: int,
    overlap_tokens: int,
) -> list[TextChunk]:
    """Chunk each page separately so every chunk maps to exactly one page for citations."""
    chunks: list[TextChunk] = []
    for page in pages:
        for text in chunk_text(
            page.text,
            count_tokens=count_tokens,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        ):
            chunks.append(
                TextChunk(
                    ordinal=len(chunks),
                    page=page.number,
                    text=text,
                    token_count=count_tokens(text),
                )
            )
    return chunks


def _tokens(window: list[Segment]) -> int:
    return sum(segment.tokens for segment in window)


def _join(window: list[Segment]) -> str:
    # The first segment's separator is dropped: a chunk never starts with whitespace.
    return window[0].text + "".join(s.separator + s.text for s in window[1:])


def _overlap_tail(window: list[Segment], overlap_tokens: int) -> list[Segment]:
    tail: list[Segment] = []
    total = 0
    for segment in reversed(window):
        if total + segment.tokens > overlap_tokens:
            break
        tail.insert(0, segment)
        total += segment.tokens
    return tail


def _fit_to_limit(sentence: str, count_tokens: TokenCounter, max_tokens: int) -> list[str]:
    """Break a sentence that is longer than a whole chunk on word boundaries."""
    if count_tokens(sentence) <= max_tokens:
        return [sentence]

    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in sentence.split():
        word_tokens = count_tokens(word)
        if current and current_tokens + word_tokens > max_tokens:
            parts.append(" ".join(current))
            current, current_tokens = [], 0
        if word_tokens > max_tokens:
            parts.extend(_split_long_word(word, count_tokens, max_tokens))
            continue
        current.append(word)
        current_tokens += word_tokens
    if current:
        parts.append(" ".join(current))
    return parts


def _split_long_word(word: str, count_tokens: TokenCounter, max_tokens: int) -> list[str]:
    # Base64 blobs, long URLs and the like. Halve until each piece fits.
    if len(word) <= 1 or count_tokens(word) <= max_tokens:
        return [word]
    middle = len(word) // 2
    return [
        *_split_long_word(word[:middle], count_tokens, max_tokens),
        *_split_long_word(word[middle:], count_tokens, max_tokens),
    ]
