import pytest

from gpt.data import load_documents


def test_load_single_text_file(tmp_path):
    file_path = tmp_path / "document.txt"
    file_path.write_text("Hello, world!", encoding="utf-8")

    documents = load_documents(file_path)

    assert len(documents) == 1
    assert documents[0][0] == file_path
    assert documents[0][1] == "Hello, world!"


def test_load_multiple_documents_recursively(tmp_path):
    first = tmp_path / "first.txt"
    first.write_text("First document.", encoding="utf-8")

    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()

    second = nested_dir / "second.txt"
    second.write_text("Second document.", encoding="utf-8")

    ignored = nested_dir / "notes.csv"
    ignored.write_text("This should be ignored.", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 2
    assert documents[0][0] == first
    assert documents[1][0] == second


def test_empty_documents_are_skipped(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")

    whitespace = tmp_path / "whitespace.txt"
    whitespace.write_text("   \n   ", encoding="utf-8")

    valid = tmp_path / "valid.txt"
    valid.write_text("Actual content.", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0][1] == "Actual content."


def test_missing_input_path_raises_error(tmp_path):
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        load_documents(missing)


def test_directory_without_valid_documents_raises_error(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(ValueError):
        load_documents(tmp_path)