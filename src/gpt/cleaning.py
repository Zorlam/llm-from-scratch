
"""Source-aware document cleaning utilities."""

import re

from bs4 import BeautifulSoup, NavigableString


HTML_UNWANTED_TAGS = [
    "script",
    "style",
    "nav",
    "footer",
    "aside",
    "noscript",
    "template",
    "svg",
    "form",
    "button",
]

HTML_BLOCK_TAGS = {
    "address",
    "article",
    "blockquote",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "main",
    "p",
    "section",
    "tr",
}

GUTENBERG_START_PATTERN = re.compile(
    r"^\s*\*{0,3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG"
    r"(?: EBOOK| EBOOK FILE)?\b.*$",
    re.IGNORECASE | re.MULTILINE,
)

GUTENBERG_END_PATTERN = re.compile(
    r"^\s*\*{0,3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG"
    r"(?: EBOOK| EBOOK FILE)?\b.*$",
    re.IGNORECASE | re.MULTILINE,
)


def clean_html(text: str) -> str:
    """Extract readable text from HTML while preserving block boundaries."""

    soup = BeautifulSoup(text, "html.parser")

    for tag in soup.find_all(HTML_UNWANTED_TAGS):
        tag.decompose()

    for tag in soup.find_all(HTML_BLOCK_TAGS):
        tag.insert_before(NavigableString("\n"))
        tag.insert_after(NavigableString("\n"))

    for tag in soup.find_all("br"):
        tag.replace_with(NavigableString("\n"))

    extracted_text = soup.get_text()

    lines = [
        line.strip()
        for line in extracted_text.splitlines()
    ]

    cleaned_lines = [
        line
        for line in lines
        if line
    ]

    return "\n\n".join(cleaned_lines)


def clean_gutenberg_text(text: str) -> str:
    """Remove Project Gutenberg headers and footers when markers exist."""

    start_match = GUTENBERG_START_PATTERN.search(text)
    end_match = GUTENBERG_END_PATTERN.search(text)

    if start_match and end_match:
        if start_match.end() >= end_match.start():
            return text

        text = text[start_match.end():end_match.start()]

    elif end_match:
        text = text[:end_match.start()]

    elif start_match:
        text = text[start_match.end():]

    return text.strip()


def clean_document(
    text: str,
    source_type: str = "plain_text",
) -> str:
    """Apply cleaning appropriate to the document's source type."""

    if source_type == "plain_text":
        return clean_gutenberg_text(text)

    if source_type == "html":
        return clean_html(text)

    if source_type == "code":
        return text

    raise ValueError(
        f"Unsupported source type: {source_type}"
    )
