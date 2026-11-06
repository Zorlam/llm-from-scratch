from pathlib import Path

from gpt.data import deduplicate_documents



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