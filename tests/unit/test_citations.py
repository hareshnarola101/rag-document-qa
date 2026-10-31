from app.qa.prompts import format_sources
from app.retrieval.citations import build_citations, make_snippet
from tests.factories import make_chunk


def test_citations_map_one_based_source_numbers_to_chunks() -> None:
    chunks = [make_chunk("alpha"), make_chunk("beta", filename="b.pdf", page=3)]

    [citation] = build_citations([2], chunks)

    assert citation.source == 2
    assert citation.chunk_id == chunks[1].chunk_id
    assert citation.document_id == chunks[1].document_id
    assert (citation.filename, citation.page, citation.snippet) == ("b.pdf", 3, "beta")


def test_invented_sources_are_dropped() -> None:
    chunks = [make_chunk("alpha")]

    assert build_citations([0, 2, -1, 99], chunks) == []


def test_repeated_sources_are_collapsed_in_citation_order() -> None:
    chunks = [make_chunk("a"), make_chunk("b"), make_chunk("c")]

    citations = build_citations([3, 1, 3, 1], chunks)

    assert [c.source for c in citations] == [3, 1]


def test_no_sources_no_citations() -> None:
    assert build_citations([], [make_chunk("a")]) == []


def test_snippet_keeps_short_text_and_collapses_whitespace() -> None:
    assert make_snippet("  hello \n  world  ") == "hello world"


def test_snippet_truncates_on_a_word_boundary() -> None:
    text = "The quick brown fox jumps over the lazy dog"

    assert make_snippet(text, max_chars=22) == "The quick brown fox…"


def test_snippet_hard_cuts_when_there_is_no_usable_space() -> None:
    assert make_snippet("a" * 50, max_chars=10) == "a" * 10 + "…"


def test_snippet_starts_at_the_sentence_that_supports_the_answer() -> None:
    text = (
        "Shifts start on Monday. SEV1 pages need a response within 15 minutes. Postmortems follow."
    )

    snippet = make_snippet(text, max_chars=50, focus="Acknowledge SEV1 pages within 15 minutes.")

    assert snippet == "…SEV1 pages need a response within 15 minutes.…"


def test_citation_snippets_follow_the_answer() -> None:
    chunks = [make_chunk("Intro sentence here. The answer is forty-two.")]

    [citation] = build_citations([1], chunks, answer="It is forty-two.")

    assert citation.snippet == "…The answer is forty-two."


def test_sources_are_numbered_with_their_location() -> None:
    chunks = [make_chunk("first", filename="a.md"), make_chunk("second", "b.pdf", page=7)]

    assert format_sources(chunks) == "[1] a.md\nfirst\n\n[2] b.pdf, page 7\nsecond"
