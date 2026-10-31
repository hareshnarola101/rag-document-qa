"""Small, dependency-free text helpers shared by the chunker, the fake models and the metrics."""

import re

_WORD_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[`A-Z0-9])")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")
# Lines that start a new block in Markdown-ish text: headings, list items, quotes, tables.
_BLOCK_START_RE = re.compile(r"^(?:#{1,6}\s|[-*+>|]\s|\d+[.)]\s)")

_STOPWORDS = """
    a an and are as at be been but by can could did do does for from had has have how i if in
    into is it its me my no not of on or our so than that the their them then there these they
    this to was we were what when where which who whom why will with would you your about after
    all also any because before being between both during each few more most other over same
    should some such through under until up very just only own s t don doesn isn aren
"""
STOPWORDS = frozenset(_STOPWORDS.split())


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _normalize(word: str) -> str:
    # Crude plural folding; good enough for lexical overlap without pulling in a stemmer.
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def content_words(text: str) -> list[str]:
    return [_normalize(w) for w in words(text) if w not in STOPWORDS]


def _logical_lines(paragraph: str) -> list[str]:
    """Re-join hard-wrapped lines (common in .txt files and PDF text) but keep list items,
    headings and table rows on their own lines."""
    lines: list[str] = []
    for raw in paragraph.split("\n"):
        line = raw.strip()
        if not line:
            continue
        starts_block = _BLOCK_START_RE.match(line) is not None
        previous_is_heading = bool(lines) and lines[-1].startswith("#")
        if lines and not starts_block and not previous_is_heading:
            lines[-1] = f"{lines[-1]} {line}"
        else:
            lines.append(line)
    return lines


def sentence_spans(text: str) -> list[tuple[str, str]]:
    """Sentences paired with the separator that came before them.

    The separator is "\n\n" at a paragraph start, "\n" at the start of a list item, heading
    or other block line, and " " between sentences on the same line. Joining sentences with
    their separators rebuilds the text with its structure intact.
    """
    spans: list[tuple[str, str]] = []
    for paragraph in _PARAGRAPH_RE.split(text):
        for line_index, line in enumerate(_logical_lines(paragraph)):
            sentences = [s.strip() for s in _SENTENCE_END_RE.split(line) if s.strip()]
            for sentence_index, sentence in enumerate(sentences):
                if sentence_index > 0:
                    separator = " "
                elif line_index > 0:
                    separator = "\n"
                else:
                    separator = "\n\n"
                spans.append((sentence, separator))
    return spans


def split_sentences(text: str) -> list[str]:
    return [sentence for sentence, _ in sentence_spans(text)]
