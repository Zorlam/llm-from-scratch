
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
        "gutenberg": False,
    },
    "frankenstein.txt": {
        "url": "https://www.gutenberg.org/cache/epub/84/pg84.txt",
        "source": "Frankenstein - Mary Shelley",
        "min_bytes": 100_000,
        "gutenberg": True,
    },
    "alice_in_wonderland.txt": {
        "url": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
        "source": "Alice's Adventures in Wonderland - Lewis Carroll",
        "min_bytes": 100_000,
        "gutenberg": True,
    },
    "sherlock_holmes.txt": {
        "url": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
        "source": "The Adventures of Sherlock Holmes - Arthur Conan Doyle",
        "min_bytes": 100_000,
        "gutenberg": True,
    },
}


# --------------------------------------------------
# Download and validation
# --------------------------------------------------

def download_dataset(url: str, min_bytes: int, gutenberg: bool) -> bytes:
    """Download a dataset and validate its contents before saving."""

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
                "The server returned an HTML page instead of plain text."
            )

        content = response.read()

    if len(content) < min_bytes:
        raise ValueError(
            f"Download is too small: {len(content):,} bytes."
        )

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(
            "Downloaded content is not valid UTF-8."
        ) from error

    if not text.strip():
        raise ValueError("Downloaded file is empty.")

    beginning = text[:2000].lower()

    if "<!doctype html" in beginning or "<html" in beginning:
        raise ValueError(
            "Downloaded content appears to be an HTML page."
        )

    if gutenberg:
        if "*** start of the project gutenberg" not in text.lower():
            raise ValueError(
                "Project Gutenberg start marker was not found."
            )

        if "*** end of the project gutenberg" not in text.lower():
            raise ValueError(
                "Project Gutenberg end marker was not found. "
                "The download may be incomplete."
            )

    return content


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
                dataset["url"],
                dataset["min_bytes"],
                dataset["gutenberg"],
            )

            output_path.write_bytes(content)

            print(
                f"Downloaded: {output_path} "
                f"({len(content):,} bytes)"
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
