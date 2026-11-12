from pathlib import Path

from gpt.data import get_language_distribution


def test_counts_documents_by_language():
    documents = [
        (
            Path("english.txt"),
            (
                "Artificial intelligence is transforming the world. "
                "Researchers are developing increasingly capable systems. "
                "These systems can understand language and process information."
            ),
        ),
        (
            Path("french.txt"),
            (
                "L'intelligence artificielle transforme le monde. "
                "Les chercheurs développent des systèmes de plus en plus "
                "performants. Ces systèmes peuvent comprendre le langage "
                "et traiter des informations."
            ),
        ),
        (
            Path("another_english.txt"),
            (
                "Machine learning allows computers to identify patterns "
                "and make predictions from data. It is widely used in "
                "science, medicine, business, and engineering."
            ),
        ),
    ]

    distribution = get_language_distribution(documents)

    assert distribution == {
        "en": 2,
        "fr": 1,
    }


def test_undetected_languages_are_counted_as_unknown():
    documents = [
        (Path("short.txt"), "Hello"),
    ]

    distribution = get_language_distribution(documents)

    assert distribution == {
        "unknown": 1,
    }


def test_source_code_is_counted_as_unknown():
    documents = [
        (
            Path("example.py"),
            "def calculate_total(price, tax): return price + tax",
        ),
    ]

    distribution = get_language_distribution(documents)

    assert distribution == {
        "unknown": 1,
    }


def test_non_english_documents_are_preserved():
    documents = [
        (
            Path("french.txt"),
            (
                "L'intelligence artificielle transforme le monde. "
                "Les chercheurs développent des systèmes de plus en plus "
                "performants. Ces systèmes peuvent comprendre le langage "
                "et traiter des informations."
            ),
        ),
    ]

    distribution = get_language_distribution(documents)

    assert sum(distribution.values()) == len(documents)
    assert distribution["fr"] == 1


def test_empty_document_list():
    distribution = get_language_distribution([])

    assert distribution == {}


def test_language_distribution_is_sorted():
    documents = [
        (
            Path("french.txt"),
            (
                "L'intelligence artificielle transforme le monde. "
                "Les chercheurs développent des systèmes de plus en plus "
                "performants. Ces systèmes peuvent comprendre le langage "
                "et traiter des informations."
            ),
        ),
        (
            Path("english.txt"),
            (
                "Artificial intelligence is transforming the world. "
                "Researchers are developing increasingly capable systems. "
                "These systems can understand language and process information."
            ),
        ),
    ]

    distribution = get_language_distribution(documents)

    assert list(distribution.keys()) == sorted(distribution.keys())