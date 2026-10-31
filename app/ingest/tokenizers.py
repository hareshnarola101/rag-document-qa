import re
from collections.abc import Callable
from functools import cache

TokenCounter = Callable[[str], int]

_REGEX_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def count_regex_tokens(text: str) -> int:
    """Words and punctuation marks. Runs offline; used by the tests and the fake provider."""
    return len(_REGEX_TOKEN_RE.findall(text))


@cache
def _tiktoken_counter() -> TokenCounter:
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    return lambda text: len(encoding.encode(text, disallowed_special=()))


def get_token_counter(name: str) -> TokenCounter:
    # cl100k is not the tokenizer of every embedding model, but it's close enough for sizing
    # chunks, and chunk_size_tokens leaves plenty of headroom under typical 512+ token limits.
    if name == "tiktoken":
        return _tiktoken_counter()
    if name == "regex":
        return count_regex_tokens
    raise ValueError(f"unknown tokenizer {name!r}")
