
"""Download public-domain literary datasets for language model training."""

import argparse
import urllib.error
import urllib.request
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

OUTPUT_DIR = Path("data/raw")

DATASETS = {
    "tinyshakespeare.txt": {
        "url": (
            "https://raw.githubusercontent.com/"
            "karpathy/char-rnn/master/data/"
            "tinyshakespeare/input.txt"
        ),
        "source": "Tiny Shakespeare",
        "min_bytes": 100_000,
        "encoding": "utf-8-sig",
        "gutenberg": False,
    },
    "frankenstein.txt": {
        "url": (
            "https://raw.githubusercontent.com/"
            "GITenberg/Frankenstein_84/master/84-0.txt"
        ),
        "source": "Frankenstein - Mary Shelley",
        "min_bytes": 100_000,
        "encoding": "utf-8-sig",
        "gutenberg": True,
        "title": "Frankenstein",
    },
    "alice_in_wonderland.txt": {
        "url": (
            "https://raw.githubusercontent.com/"
            "GITenberg/Alice-s-Adventures-in-Wonderland_11/"
            "master/11-0.txt"
        ),
        "source": "Alice's Adventures in Wonderland - Lewis Carroll",
        "min_bytes": 100_000,
        "encoding": "utf-8-sig",
        "gutenberg": True,
        "title": "Alice's Adventures in Wonderland",
    },
    "sherlock_holmes.txt": {
        "url": (
            "https://raw.githubusercontent.com/"
            "GITenberg/The-Adventures-of-Sherlock-Holmes_1661/"
            "master/1661-8.txt"
        ),
        "source": "The Adventures of Sherlock Holmes - Arthur Conan Doyle",
        "min_bytes": 100_000,
        "encoding": "cp1252",
        "gutenberg": True,
        "title": "The Adventures of Sherlock Holmes",
    },
}


# --------------------------------------------------
# Validation helpers
# --------------------------------------------------

def normalize_text(text: str) -> str:
    """Normalize apostrophes for reliable title comparisons."""
    return (
        text.casefold()
        .replace("’", "'")
        .replace("‘", "'")
    )


def validate_gutenberg(text: str, expected_title: str) -> None:
    """Validate modern and legacy Project Gutenberg text formats."""

    normalized = normalize_text(text)

    start_marker = "*** start of the project gutenberg"
    end_marker = "*** end of the project gutenberg"

    has_start_marker = start_marker in normalized
    has_end_marker = end_marker in normalized

    if has_start_marker:
        if not has_end_marker:
            raise ValueError(
                "Gutenberg start marker exists, but the end marker "
                "is missing. The download may be incomplete."
            )
        return

    # Older GITenberg editions use a different header/footer format.
    header = normalize_text(text[:2000])
    footer = normalize_text(text[-2000:])
    title = normalize_text(expected_title)

    if "project gutenberg" not in header:
        raise ValueError("Project Gutenberg header was not found.")

    if title not in header:
        raise ValueError(
            f"Expected book title was not found in the header: "
            f"{expected_title}"
        )

    if "most people start at our web site" not in footer:
        raise ValueError(
            "Legacy Gutenberg footer was not found. "
            "The download may be incomplete."
        )


# --------------------------------------------------
# Download and validation
# --------------------------------------------------

def download_dataset(
    url: str,
    min_bytes: int,
    encoding: str,
    gutenberg: bool,
    title: str | None = None,
) -> bytes:
    """Download, decode, validate, and return UTF-8 content."""

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "LLM-From-Scratch-DatasetDownloader/1.0"
        },
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status != 200:
            raise ValueError(
                f"Unexpected HTTP status: {response.status}"
            )

        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        if "text/html" in content_type:
            raise ValueError(
                "The server returned HTML instead of plain text."
            )

        content = response.read()

    if len(content) < min_bytes:
        raise ValueError(
            f"Download is too small: {len(content):,} bytes."
        )

    try:
        text = content.decode(encoding)
    except UnicodeDecodeError as error:
        raise ValueError(
            f"Downloaded content is not valid {encoding}."
        ) from error

    if not text.strip():
        raise ValueError("Downloaded file is empty.")

    beginning = text[:2000].casefold()

    if "<!doctype html" in beginning or "<html" in beginning:
        raise ValueError(
            "Downloaded content appears to be an HTML page."
        )

    if "�" in text:
        raise ValueError(
            "Decoded content contains replacement characters."
        )

    if gutenberg:
        if title is None:
            raise ValueError(
                "A title is required for Gutenberg validation."
            )

        validate_gutenberg(text, title)

    # Store all datasets consistently as UTF-8.
    return text.encode("utf-8")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download literary datasets for language model training."
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Download files again, even if they already exist.",
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    successful = 0
    skipped = 0
    failed = 0

    for filename, dataset in DATASETS.items():
        output_path = OUTPUT_DIR / filename

        print(f"\nDataset: {dataset['source']}")

        if output_path.exists() and not args.force:
            print(
                f"Already exists: {output_path} "
                f"({output_path.stat().st_size:,} bytes)"
            )
            skipped += 1
            continue

        try:
            content = download_dataset(
                url=dataset["url"],
                min_bytes=dataset["min_bytes"],
                encoding=dataset["encoding"],
                gutenberg=dataset["gutenberg"],
                title=dataset.get("title"),
            )

            output_path.write_bytes(content)

            print(
                f"Downloaded: {output_path} "
                f"({len(content):,} bytes, UTF-8)"
            )

            successful += 1

        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            ValueError,
        ) as error:
            print(f"Download failed: {error}")
            failed += 1

    print("\n" + "=" * 45)
    print("DOWNLOAD SUMMARY")
    print("=" * 45)
    print(f"Downloaded: {successful}")
    print(f"Skipped:    {skipped}")
    print(f"Failed:     {failed}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
