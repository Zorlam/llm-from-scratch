from gpt.data import load_documents


def test_load_mixed_source_directory(tmp_path):
    # Plain-text document
    text_file = tmp_path / "article.txt"
    text_file.write_text(
        "First paragraph.\r\n\r\n\r\nSecond paragraph.   ",
        encoding="utf-8",
    )

    # HTML document
    html_file = tmp_path / "website.html"
    html_file.write_text(
        """
        <html>
            <body>
                <nav>Home | About | Contact</nav>
                <main>
                    <h1>Machine Learning</h1>
                    <p>Machine learning is a branch of AI.</p>
                </main>
                <footer>Copyright 2026</footer>
            </body>
        </html>
        """,
        encoding="utf-8",
    )

    # Python source file
    code_file = tmp_path / "example.py"
    code = "def greet():\n    print('Hello')\n\n"
    code_file.write_text(code, encoding="utf-8")

    # Unsupported file should be ignored during directory loading.
    json_file = tmp_path / "metadata.json"
    json_file.write_text('{"author": "Example"}', encoding="utf-8")

    documents = load_documents(tmp_path)

    results = {
        path.name: text
        for path, text in documents
    }

    assert len(documents) == 3

    # Plain text is normalized.
    assert results["article.txt"] == (
        "First paragraph.\n\nSecond paragraph."
    )

    # HTML is extracted and cleaned.
    assert "Machine Learning" in results["website.html"]
    assert "Machine learning is a branch of AI." in results["website.html"]
    assert "Home | About | Contact" not in results["website.html"]
    assert "Copyright 2026" not in results["website.html"]

    # Code formatting is preserved exactly.
    assert results["example.py"] == code

    # Unsupported files are not loaded.
    assert "metadata.json" not in results


def test_load_single_html_file(tmp_path):
    html_file = tmp_path / "article.html"
    html_file.write_text(
        "<article><h1>Introduction</h1>"
        "<p>This is an article.</p></article>",
        encoding="utf-8",
    )

    documents = load_documents(html_file)

    assert len(documents) == 1
    assert "Introduction" in documents[0][1]
    assert "This is an article." in documents[0][1]


def test_unsupported_single_file_raises_error(tmp_path):
    json_file = tmp_path / "data.json"
    json_file.write_text(
        '{"message": "Hello"}',
        encoding="utf-8",
    )

    import pytest

    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_documents(json_file)