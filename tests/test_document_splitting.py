from pathlib import Path

import pytest

from gpt.data import split_documents


def test_split_documents_keeps_documents_separate():
    documents = [
        (Path("first.txt"), "First document."),
        (Path("second.txt"), "Second document."),
        (Path("third.txt"), "Third document."),
        (Path("fourth.txt"), "Fourth document."),
    ]

    train, val = split_documents(
        documents,
        val_fraction=0.25,
        seed=42,
    )

    assert len(train) == 3
    assert len(val) == 1

    assert set(train).isdisjoint(set(val))
    assert set(train + val) == {
        "First document.",
        "Second document.",
        "Third document.",
        "Fourth document.",
    }


def test_split_documents_is_reproducible():
    documents = [
        (Path(f"document_{i}.txt"), f"Document {i}.")
        for i in range(10)
    ]

    first_split = split_documents(
        documents,
        val_fraction=0.2,
        seed=42,
    )

    second_split = split_documents(
        documents,
        val_fraction=0.2,
        seed=42,
    )

    assert first_split == second_split


def test_split_documents_requires_two_documents():
    documents = [
        (Path("only.txt"), "Only document.")
    ]

    with pytest.raises(ValueError):
        split_documents(documents)


def test_split_documents_rejects_invalid_fraction():
    documents = [
        (Path("first.txt"), "First."),
        (Path("second.txt"), "Second."),
    ]

    with pytest.raises(ValueError):
        split_documents(documents, val_fraction=1.0)