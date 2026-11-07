from pathlib import Path

from gpt.data import (
    deduplicate_documents,
    deduplicate_near_duplicates,
)


def test_duplicate_documents_are_removed():
    documents = [
        (Path("first.txt"), "Hello world."),
        (Path("second.txt"), "Another document."),
        (Path("third.txt"), "Hello world."),
        (Path("fourth.txt"), "A different document."),
    ]

    unique, removed = deduplicate_documents(documents)

    assert len(unique) == 3
    assert removed == 1

    assert [text for _, text in unique] == [
        "Hello world.",
        "Another document.",
        "A different document.",
    ]


def test_unique_documents_are_preserved():
    documents = [
        (Path("first.txt"), "First document."),
        (Path("second.txt"), "Second document."),
        (Path("third.txt"), "Third document."),
    ]

    unique, removed = deduplicate_documents(documents)

    assert unique == documents
    assert removed == 0


def test_empty_document_list():
    unique, removed = deduplicate_documents([])

    assert unique == []
    assert removed == 0


def test_identical_content_with_different_paths_is_duplicate():
    documents = [
        (Path("folder_a/article.txt"), "Same article."),
        (Path("folder_b/article.txt"), "Same article."),
    ]

    unique, removed = deduplicate_documents(documents)

    assert len(unique) == 1
    assert unique[0][0] == Path("folder_a/article.txt")
    assert removed == 1


def test_near_duplicate_documents_are_removed():
    documents = [
        (
            Path("first.txt"),
            "Artificial intelligence is transforming healthcare "
            "by improving diagnosis and patient care across hospitals.",
        ),
        (
            Path("second.txt"),
            "Artificial intelligence is transforming healthcare "
            "by improving diagnosis and patient care across hospitals "
            "around the world.",
        ),
        (
            Path("third.txt"),
            "Python is a programming language used for web development "
            "automation and data science.",
        ),
    ]

    unique, removed = deduplicate_near_duplicates(documents)

    assert len(unique) == 2
    assert removed == 1


def test_completely_different_documents_are_preserved():
    documents = [
        (
            Path("first.txt"),
            "Python programming involves writing software applications.",
        ),
        (
            Path("second.txt"),
            "Agriculture involves cultivating crops and raising livestock.",
        ),
    ]

    unique, removed = deduplicate_near_duplicates(documents)

    assert unique == documents
    assert removed == 0


def test_short_documents_are_not_removed():
    documents = [
        (Path("first.txt"), "Hello world."),
        (Path("second.txt"), "Hello there."),
    ]

    unique, removed = deduplicate_near_duplicates(documents)

    assert unique == documents
    assert removed == 0


def test_near_duplicate_empty_document_list():
    unique, removed = deduplicate_near_duplicates([])

    assert unique == []
    assert removed == 0