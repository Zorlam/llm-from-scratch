"""Source-aware document cleaning utilities."""

from bs4 import BeautifulSoup, NavigableString
from bs4.element import Tag


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


def clean_html(text: str) -> str:
    """Extract readable text from HTML while preserving block boundaries."""

    soup = BeautifulSoup(text, "html.parser")

    # Remove elements that generally contain non-article content.
    for tag in soup.find_all(HTML_UNWANTED_TAGS):
        tag.decompose()

    # Preserve paragraph and heading boundaries.
    for tag in soup.find_all(HTML_BLOCK_TAGS):
        tag.insert_before(NavigableString("\n"))
        tag.insert_after(NavigableString("\n"))

    # Preserve explicit line breaks.
    for tag in soup.find_all("br"):
        tag.replace_with(NavigableString("\n"))

    extracted_text = soup.get_text()

    # Normalize horizontal whitespace without destroying indentation.
    lines = [
        line.strip()
        for line in extracted_text.splitlines()
    ]

    # Remove empty lines and excessive whitespace.
    cleaned_lines = [
        line
        for line in lines
        if line
    ]

    return "\n\n".join(cleaned_lines)


def clean_document(
    text: str,
    source_type: str = "plain_text",
) -> str:
    """Apply cleaning appropriate to the document's source type."""

    if source_type == "plain_text":
        return text

    if source_type == "html":
        return clean_html(text)

    if source_type == "code":
        # Source code must retain its original formatting.
        return text

    raise ValueError(
        f"Unsupported source type: {source_type}"
    )