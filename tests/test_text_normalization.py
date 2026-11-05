from gpt.data import normalize_text


def test_removes_utf8_bom():
    text = "\ufeffHello world."

    assert normalize_text(text) == "Hello world."


def test_normalizes_line_endings():
    text = "First line\r\nSecond line\rThird line"

    expected = "First line\nSecond line\nThird line"

    assert normalize_text(text) == expected


def test_removes_null_bytes():
    text = "Hello\x00 world."

    assert normalize_text(text) == "Hello world."


def test_preserves_tabs_and_newlines():
    text = "First line\twith tab.\nSecond line."

    assert normalize_text(text) == text


def test_removes_trailing_whitespace():
    text = "First line   \nSecond line\t\nThird line   "

    expected = "First line\nSecond line\nThird line"

    assert normalize_text(text) == expected


def test_collapses_excessive_blank_lines():
    text = "First paragraph.\n\n\n\nSecond paragraph."

    expected = "First paragraph.\n\nSecond paragraph."

    assert normalize_text(text) == expected


def test_preserves_leading_indentation():
    text = "def hello():\n    print('Hello')"

    assert normalize_text(text) == text


def test_preserves_unicode():
    text = "Café — naïve résumé. 你好."

    assert normalize_text(text) == text


def test_removes_other_control_characters():
    text = "Hello\x01\x02 world."

    assert normalize_text(text) == "Hello world."


def test_empty_text_remains_empty():
    assert normalize_text("") == ""