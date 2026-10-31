import pytest

from app.ingest.parsers import (
    DocumentParseError,
    UnsupportedFileTypeError,
    content_type_for,
    normalize_whitespace,
    parse_document,
)
from tests.pdf_factory import make_pdf


def test_pdf_keeps_page_numbers_and_skips_blank_pages() -> None:
    data = make_pdf(["First page text.", "", "Third page\nhas two lines."])

    pages = parse_document("report.pdf", data)

    assert [p.number for p in pages] == [1, 3]
    assert pages[0].text == "First page text."
    assert "has two lines." in pages[1].text


def test_markdown_and_text_have_no_page_numbers() -> None:
    [page] = parse_document("notes.MD", b"# Title\n\nBody text.")

    assert page.number is None
    assert page.text == "# Title\n\nBody text."


def test_non_utf8_text_falls_back_to_latin1() -> None:
    [page] = parse_document("legacy.txt", "café".encode("latin-1"))

    assert page.text == "café"


def test_whitespace_only_document_yields_nothing() -> None:
    assert parse_document("empty.txt", b" \n\t\n ") == []


def test_unsupported_extension_is_rejected() -> None:
    with pytest.raises(UnsupportedFileTypeError, match=r"\.docx"):
        content_type_for("contract.docx")


def test_corrupt_pdf_raises_parse_error() -> None:
    with pytest.raises(DocumentParseError):
        parse_document("broken.pdf", b"%PDF-1.4 this is not really a pdf")


def test_normalize_whitespace() -> None:
    raw = "a  \t b\r\n\r\n\r\n\r\nc   \n"

    assert normalize_whitespace(raw) == "a b\n\nc"
