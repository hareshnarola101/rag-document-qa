import itertools

import pytest

from app.ingest.chunking import chunk_pages, chunk_text
from app.ingest.parsers import ParsedPage
from app.ingest.tokenizers import count_regex_tokens


def numbered_sentences(count: int) -> str:
    # Each sentence is exactly 6 regex tokens: "Sentence number 7 is here."
    return " ".join(f"Sentence number {i} is here." for i in range(count))


def chunk(text: str, max_tokens: int = 30, overlap_tokens: int = 6) -> list[str]:
    return chunk_text(
        text, count_tokens=count_regex_tokens, max_tokens=max_tokens, overlap_tokens=overlap_tokens
    )


def test_short_text_is_a_single_chunk() -> None:
    assert chunk("One sentence. Another one.") == ["One sentence. Another one."]


def test_chunks_keep_paragraphs_and_list_items() -> None:
    text = "# Title\n\nFirst paragraph. Still first.\n\n- item one\n- item two"

    assert chunk(text, max_tokens=100) == [text]


def test_empty_text_produces_no_chunks() -> None:
    assert chunk("   \n\n ") == []


def test_chunks_respect_the_token_limit() -> None:
    chunks = chunk(numbered_sentences(40), max_tokens=30)

    assert len(chunks) > 1
    assert all(count_regex_tokens(c) <= 30 for c in chunks)


def test_consecutive_chunks_overlap_by_whole_sentences() -> None:
    chunks = chunk(numbered_sentences(20), max_tokens=30, overlap_tokens=6)

    for previous, current in itertools.pairwise(chunks):
        last_sentence = "Sentence" + previous.rsplit("Sentence", 1)[1]
        assert current.startswith(last_sentence)


def test_zero_overlap_partitions_the_text() -> None:
    text = numbered_sentences(20)
    chunks = chunk(text, max_tokens=30, overlap_tokens=0)

    assert " ".join(chunks) == text


def test_sentence_longer_than_the_limit_is_split_on_words() -> None:
    long_sentence = " ".join(f"word{i}" for i in range(100)) + "."
    chunks = chunk(long_sentence, max_tokens=25, overlap_tokens=0)

    assert all(count_regex_tokens(c) <= 25 for c in chunks)
    assert " ".join(chunks) == long_sentence


def test_unbreakable_word_is_sliced() -> None:
    blob = "x" * 50 + "-" * 50  # 51 regex tokens, no spaces
    chunks = chunk(blob, max_tokens=20, overlap_tokens=0)

    assert "".join(c.replace(" ", "") for c in chunks) == blob
    assert all(count_regex_tokens(c) <= 20 for c in chunks)


@pytest.mark.parametrize(("max_tokens", "overlap"), [(0, 0), (10, 10), (10, -1)])
def test_invalid_limits_are_rejected(max_tokens: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="tokens"):
        chunk("text", max_tokens=max_tokens, overlap_tokens=overlap)


def test_chunk_pages_keeps_page_numbers_and_global_ordinals() -> None:
    pages = [ParsedPage(1, numbered_sentences(12)), ParsedPage(2, "Only one sentence.")]

    chunks = chunk_pages(pages, count_tokens=count_regex_tokens, max_tokens=30, overlap_tokens=6)

    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert {c.page for c in chunks[:-1]} == {1}
    assert chunks[-1].page == 2
    assert chunks[-1].text == "Only one sentence."
    assert all(c.token_count == count_regex_tokens(c.text) for c in chunks)
