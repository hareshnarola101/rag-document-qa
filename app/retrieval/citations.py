from collections.abc import Sequence

from app.core.text import content_words, split_sentences
from app.retrieval.types import RetrievedChunk
from app.schemas import Citation


def make_snippet(text: str, max_chars: int = 240, focus: str = "") -> str:
    """A short excerpt of `text`, starting at the sentence that best matches `focus`.

    Chunks are a few hundred tokens long, so starting the snippet where the supporting
    sentence is makes the citation useful at a glance.
    """
    start = _best_sentence_start(text, focus)
    prefix = "…" if start > 0 else ""
    text = " ".join(text[start:].split())
    if len(text) <= max_chars:
        return prefix + text
    cut = text[:max_chars]
    # Prefer ending on a word boundary unless that would throw away most of the snippet.
    space = cut.rfind(" ")
    if space > max_chars // 2:
        cut = cut[:space]
    return prefix + cut.rstrip(" ,;:") + "…"


def _best_sentence_start(text: str, focus: str) -> int:
    terms = set(content_words(focus))
    best_score, best_start = 0, 0
    if not terms:
        return best_start
    for sentence in split_sentences(text):
        score = len(terms & set(content_words(sentence)))
        position = text.find(sentence)
        if score > best_score and position >= 0:
            best_score, best_start = score, position
    return best_start


def build_citations(
    cited_sources: Sequence[int],
    chunks: Sequence[RetrievedChunk],
    *,
    answer: str = "",
    max_snippet_chars: int = 240,
) -> list[Citation]:
    """Turn the model's [n] source numbers into citations.

    Numbers are 1-based positions in `chunks`, matching how the prompt labels sources.
    Out-of-range numbers (the model inventing a source) are dropped, and repeats are
    collapsed while keeping the order the model cited them in.
    """
    citations: list[Citation] = []
    for number in dict.fromkeys(cited_sources):
        if not 1 <= number <= len(chunks):
            continue
        chunk = chunks[number - 1]
        citations.append(
            Citation(
                source=number,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page=chunk.page,
                snippet=make_snippet(chunk.content, max_snippet_chars, focus=answer),
            )
        )
    return citations
