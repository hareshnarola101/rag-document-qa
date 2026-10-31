from app.core.text import content_words, sentence_spans, split_sentences


def test_split_sentences_rejoins_hard_wrapped_lines() -> None:
    text = "The event loop runs one\ncoroutine at a time. It yields\nonly at an await."

    assert split_sentences(text) == [
        "The event loop runs one coroutine at a time.",
        "It yields only at an await.",
    ]


def test_split_sentences_keeps_markdown_blocks_separate() -> None:
    text = "# Title\nIntro line.\n\n- first item\n- second item\n1. numbered"

    assert split_sentences(text) == [
        "# Title",
        "Intro line.",
        "- first item",
        "- second item",
        "1. numbered",
    ]


def test_sentence_spans_record_the_separator_before_each_sentence() -> None:
    text = "Title line\n\nOne. Two.\n- item"

    assert sentence_spans(text) == [
        ("Title line", "\n\n"),
        ("One.", "\n\n"),
        ("Two.", " "),
        ("- item", "\n"),
    ]


def test_split_sentences_does_not_split_lowercase_continuations() -> None:
    assert split_sentences("Use e.g. a semaphore here.") == ["Use e.g. a semaphore here."]


def test_content_words_drop_stopwords_and_fold_plurals() -> None:
    assert content_words("How do the caches handle requests?") == ["cache", "handle", "request"]
