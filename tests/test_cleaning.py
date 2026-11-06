import pytest

from gpt.cleaning import clean_html, clean_document


def test_clean_html_removes_unwanted_elements():
    html = """
    <html>
        <body>
            <nav>Home | About | Contact</nav>
            <main>
                <h1>Artificial Intelligence</h1>
                <p>AI is a field of computer science.</p>
            </main>
            <footer>Copyright 2026</footer>
            <script>alert("Hello");</script>
            <style>body { color: red; }</style>
        </body>
    </html>
    """

    result = clean_html(html)

    assert "Artificial Intelligence" in result
    assert "AI is a field of computer science." in result

    assert "Home | About | Contact" not in result
    assert "Copyright 2026" not in result
    assert "alert" not in result
    assert "color: red" not in result


def test_clean_html_preserves_paragraph_boundaries():
    html = """
    <article>
        <p>First paragraph.</p>
        <p>Second paragraph.</p>
    </article>
    """

    result = clean_html(html)

    assert "First paragraph." in result
    assert "Second paragraph." in result
    assert "\n\n" in result


def test_clean_html_preserves_line_breaks():
    html = "<p>First line.<br>Second line.</p>"

    result = clean_html(html)

    assert "First line." in result
    assert "Second line." in result
    assert "\n" in result


def test_plain_text_is_preserved():
    text = "First paragraph.\n\nSecond paragraph."

    assert clean_document(text, "plain_text") == text


def test_code_is_preserved_exactly():
    code = "def greet():\n    print('Hello')\n"

    assert clean_document(code, "code") == code


def test_html_source_uses_html_cleaner():
    html = "<p>Hello world.</p>"

    result = clean_document(html, "html")

    assert "Hello world." in result
    assert "<p>" not in result


def test_unsupported_source_type_raises_error():
    with pytest.raises(ValueError):
        clean_document("Some text", "pdf")