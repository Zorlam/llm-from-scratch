from pathlib import Path

from gpt.data import filter_low_quality_documents


def test_short_documents_are_removed():
    documents = [
        (Path("short.txt"), "This document has only five words."),
        (
            Path("long.txt"),
            "This is a sufficiently long document containing "
            "more than twenty words to pass the minimum length "
            "requirement without any difficulty.",
        ),
    ]

    unique, removed = filter_low_quality_documents(documents)

    assert len(unique) == 1
    assert unique[0][0] == Path("long.txt")
    assert removed == 1


def test_documents_with_repeated_lines_are_removed():
    repeated_text = (
        "This is a useful line of text.\n"
        "This is another useful line of text.\n"
        "This is a useful line of text.\n"
        "This is another useful line of text.\n"
        "This is a useful line of text.\n"
        "This is another useful line of text.\n"
    )

    documents = [
        (Path("repeated.txt"), repeated_text),
    ]

    unique, removed = filter_low_quality_documents(
        documents,
        min_words=0,
    )

    assert unique == []
    assert removed == 1


def test_high_quality_documents_are_preserved():
    documents = [
        (
            Path("article.txt"),
            "Artificial intelligence is changing many industries. "
            "Researchers are developing increasingly capable systems. "
            "These systems can process information, generate language, "
            "and assist people with complex tasks. "
            "Careful research remains important for future progress.",
        ),
    ]

    unique, removed = filter_low_quality_documents(documents)

    assert unique == documents
    assert removed == 0


def test_empty_document_list():
    unique, removed = filter_low_quality_documents([])

    assert unique == []
    assert removed == 0


def test_minimum_word_threshold_is_configurable():
    documents = [
        (Path("document.txt"), "This document contains exactly six words."),
    ]

    unique, removed = filter_low_quality_documents(
        documents,
        min_words=5,
    )

    assert unique == documents
    assert removed == 0