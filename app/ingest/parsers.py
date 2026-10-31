import io
import re
from dataclasses import dataclass
from pathlib import PurePath

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class UnsupportedFileTypeError(ValueError):
    pass


class DocumentParseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ParsedPage:
    number: int | None  # 1-based for PDFs; None for formats without pages
    text: str


CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
}

_HORIZONTAL_SPACE_RE = re.compile(r"[ \t\f\v]+")
_EXTRA_BLANK_LINES_RE = re.compile(r"\n{3,}")


def content_type_for(filename: str) -> str:
    suffix = PurePath(filename).suffix.lower()
    try:
        return CONTENT_TYPES[suffix]
    except KeyError:
        supported = ", ".join(sorted(CONTENT_TYPES))
        raise UnsupportedFileTypeError(
            f"unsupported file type {suffix or '(none)'!r}; expected one of {supported}"
        ) from None


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_HORIZONTAL_SPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    return _EXTRA_BLANK_LINES_RE.sub("\n\n", "\n".join(lines)).strip()


def parse_document(filename: str, data: bytes) -> list[ParsedPage]:
    """Extract text, keeping page numbers for PDFs so answers can cite them."""
    if content_type_for(filename) == "application/pdf":
        pages = _parse_pdf(data)
    else:
        pages = [ParsedPage(number=None, text=_decode_text(data))]

    pages = [ParsedPage(p.number, normalize_whitespace(p.text)) for p in pages]
    return [p for p in pages if p.text]


def _parse_pdf(data: bytes) -> list[ParsedPage]:
    try:
        reader = PdfReader(io.BytesIO(data))
        return [
            ParsedPage(number=index, text=page.extract_text() or "")
            for index, page in enumerate(reader.pages, start=1)
        ]
    except PdfReadError as exc:
        raise DocumentParseError(f"could not read PDF: {exc}") from exc


def _decode_text(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Latin-1 never fails, which beats rejecting a file over one stray byte.
        return data.decode("latin-1")
