"""Dataset loading, preparation, and batch sampling."""

import hashlib
import re
import unicodedata
from pathlib import Path

import numpy as np
from datasketch import MinHash, MinHashLSH

from gpt.cleaning import clean_document
from gpt.tokenizer import BPETokenizer


# Supported file extensions and their source types.
SUPPORTED_EXTENSIONS = {
    ".txt": "plain_text",
    ".md": "plain_text",
    ".rst": "plain_text",
    ".html": "html",
    ".htm": "html",
    ".py": "code",
    ".js": "code",
    ".ts": "code",
    ".jsx": "code",
    ".tsx": "code",
    ".java": "code",
    ".cpp": "code",
    ".c": "code",
    ".h": "code",
    ".hpp": "code",
    ".go": "code",
    ".rs": "code",
    ".sql": "code",
}


def normalize_text(text: str) -> str:
    """Normalize text while preserving meaningful formatting."""

    text = text.lstrip("\ufeff")

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = "".join(
        character
        for character in text
        if character in "\n\t"
        or unicodedata.category(character) != "Cc"
    )

    text = re.sub(r"[ \t]+(?=\n)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.rstrip(" \t\n")

    return text


def load_documents(input_path: str | Path) -> list[tuple[Path, str]]:
    """Load and clean supported documents from a file or directory."""

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input path does not exist: {input_path}"
        )

    if input_path.is_file():
        files = [input_path]

    elif input_path.is_dir():
        files = sorted(
            file_path
            for file_path in input_path.rglob("*")
            if file_path.is_file()
            and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    else:
        raise ValueError(
            f"Input path must be a file or directory: {input_path}"
        )

    documents = []

    for file_path in files:
        source_type = SUPPORTED_EXTENSIONS.get(
            file_path.suffix.lower()
        )

        if source_type is None:
            raise ValueError(
                f"Unsupported file extension: {file_path.suffix}"
            )

        text = file_path.read_text(encoding="utf-8")

        text = clean_document(
            text,
            source_type=source_type,
        )

        if source_type != "code":
            text = normalize_text(text)

        if not text.strip():
            continue

        documents.append((file_path, text))

    if not documents:
        raise ValueError(
            f"No non-empty supported documents found in: {input_path}"
        )

    return documents


def deduplicate_documents(
    documents: list[tuple[Path, str]],
) -> tuple[list[tuple[Path, str]], int]:
    """Remove exact duplicate documents using SHA-256 hashes."""

    seen_hashes = set()
    unique_documents = []

    for document in documents:
        path, text = document

        document_hash = hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

        if document_hash in seen_hashes:
            continue

        seen_hashes.add(document_hash)
        unique_documents.append((path, text))

    duplicates_removed = len(documents) - len(unique_documents)

    return unique_documents, duplicates_removed


def _create_shingles(
    text: str,
    shingle_size: int,
) -> set[tuple[str, ...]]:
    """Create word-based shingles from a document."""

    words = re.findall(r"\w+", text.casefold())

    return {
        tuple(words[index:index + shingle_size])
        for index in range(len(words) - shingle_size + 1)
    }


def _create_minhash(
    shingles: set[tuple[str, ...]],
    num_perm: int,
) -> MinHash:
    """Create a MinHash signature from document shingles."""

    minhash = MinHash(num_perm=num_perm)

    for shingle in sorted(shingles):
        minhash.update(" ".join(shingle).encode("utf-8"))

    return minhash


def deduplicate_near_duplicates(
    documents: list[tuple[Path, str]],
    threshold: float = 0.75,
    shingle_size: int = 3,
    num_perm: int = 128,
    min_words: int = 5,
) -> tuple[list[tuple[Path, str]], int]:
    """Remove near-duplicate documents using MinHash and LSH."""

    if not 0 < threshold <= 1:
        raise ValueError(
            "threshold must be greater than 0 and at most 1."
        )

    if shingle_size < 1:
        raise ValueError(
            "shingle_size must be at least 1."
        )

    if num_perm < 16:
        raise ValueError(
            "num_perm must be at least 16."
        )

    if min_words < shingle_size:
        raise ValueError(
            "min_words cannot be smaller than shingle_size."
        )

    if not documents:
        return [], 0

    # Retrieve a broader candidate set, then verify using the
    # original Jaccard similarity threshold below.
    lsh = MinHashLSH(
        threshold=max(0.1, threshold - 0.2),
        num_perm=num_perm,
    )

    unique_documents = []
    unique_shingles = {}
    duplicates_removed = 0

    for document in documents:
        path, text = document

        words = re.findall(r"\w+", text.casefold())

        if len(words) < min_words:
            unique_documents.append((path, text))
            continue

        shingles = _create_shingles(
            text,
            shingle_size,
        )

        if not shingles:
            unique_documents.append((path, text))
            continue

        minhash = _create_minhash(
            shingles,
            num_perm,
        )

        candidates = lsh.query(minhash)

        is_duplicate = False

        for candidate_key in candidates:
            candidate_shingles = unique_shingles[candidate_key]

            intersection = len(
                shingles & candidate_shingles
            )

            union = len(
                shingles | candidate_shingles
            )

            similarity = intersection / union

            if similarity >= threshold:
                is_duplicate = True
                break

        if is_duplicate:
            duplicates_removed += 1
            continue

        document_key = str(len(unique_documents))

        lsh.insert(
            document_key,
            minhash,
        )

        unique_shingles[document_key] = shingles
        unique_documents.append((path, text))

    return unique_documents, duplicates_removed


def filter_low_quality_documents(
    documents: list[tuple[Path, str]],
    min_words: int = 20,
    max_repeated_line_ratio: float = 0.5,
) -> tuple[list[tuple[Path, str]], int]:
    """
    Remove documents that are too short or excessively repetitive.

    A repeated line is a non-empty line that has appeared previously
    in the same document. Repetition is calculated as the proportion
    of non-empty lines that are repeated.
    """

    if min_words < 0:
        raise ValueError(
            "min_words cannot be negative."
        )

    if not 0 <= max_repeated_line_ratio <= 1:
        raise ValueError(
            "max_repeated_line_ratio must be between 0 and 1."
        )

    unique_documents = []
    documents_removed = 0

    for path, text in documents:
        words = re.findall(r"\w+", text)

        if len(words) < min_words:
            documents_removed += 1
            continue

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if lines:
            seen_lines = set()
            repeated_lines = 0

            for line in lines:
                if line in seen_lines:
                    repeated_lines += 1
                else:
                    seen_lines.add(line)

            repeated_line_ratio = (
                repeated_lines / len(lines)
            )

            if repeated_line_ratio > max_repeated_line_ratio:
                documents_removed += 1
                continue

        unique_documents.append((path, text))

    return unique_documents, documents_removed


def split_documents(
    documents: list[tuple[Path, str]],
    val_fraction: float = 0.1,
    seed: int = 1337,
) -> tuple[list[str], list[str]]:
    """Split documents into training and validation sets."""

    if not 0 < val_fraction < 1:
        raise ValueError(
            "val_fraction must be between 0 and 1."
        )

    if len(documents) < 2:
        raise ValueError(
            "At least two documents are required for a document-level split."
        )

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(documents))

    val_count = max(1, round(len(documents) * val_fraction))
    val_count = min(val_count, len(documents) - 1)

    val_indices = set(indices[:val_count])

    train_texts = [
        text
        for index, (_, text) in enumerate(documents)
        if index not in val_indices
    ]

    val_texts = [
        text
        for index, (_, text) in enumerate(documents)
        if index in val_indices
    ]

    return train_texts, val_texts


def encode_documents(
    documents: list[str],
    tokenizer: BPETokenizer,
) -> np.ndarray:
    """Encode documents and append an end-of-document token to each."""

    token_ids = []

    for document in documents:
        token_ids.extend(tokenizer.encode(document))
        token_ids.append(tokenizer.end_of_document_id)

    return np.array(token_ids, dtype=np.uint16)


def prepare_data(
    raw_path: str | Path,
    out_dir: str | Path,
    val_fraction: float = 0.1,
    vocab_size: int = 8000,
    min_frequency: int = 2,
    seed: int = 1337,
) -> dict:
    """Prepare a normalized, deduplicated BPE-tokenized dataset."""

    if not 0 < val_fraction < 1:
        raise ValueError(
            "val_fraction must be between 0 and 1."
        )

    documents = load_documents(raw_path)

    original_document_count = len(documents)

    documents, duplicates_removed = deduplicate_documents(
        documents
    )

    documents, near_duplicates_removed = deduplicate_near_duplicates(
        documents
    )

    documents, low_quality_removed = filter_low_quality_documents(
        documents
    )

    if not documents:
        raise ValueError(
            "No documents remain after duplicate and quality filtering."
        )

    if len(documents) == 1:
        text = documents[0][1]

        split = int(len(text) * (1 - val_fraction))

        train_texts = [text[:split]]
        val_texts = [text[split:]]

        if not train_texts[0] or not val_texts[0]:
            raise ValueError(
                "Document is too short to create training and validation splits."
            )

        split_method = "character"

    else:
        train_texts, val_texts = split_documents(
            documents,
            val_fraction=val_fraction,
            seed=seed,
        )

        split_method = "document"

    print("Training BPE tokenizer...")

    tokenizer = BPETokenizer.train(
        train_texts,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )

    train_ids = encode_documents(train_texts, tokenizer)
    val_ids = encode_documents(val_texts, tokenizer)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")
    tokenizer.save(out_dir / "tokenizer.json")

    return {
        "documents": original_document_count,
        "unique_documents": len(documents),
        "duplicates_removed": duplicates_removed,
        "near_duplicates_removed": near_duplicates_removed,
        "low_quality_removed": low_quality_removed,
        "split_method": split_method,
        "characters": sum(len(text) for _, text in documents),
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "train_documents": len(train_texts),
        "val_documents": len(val_texts),
    }


def get_batch(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    device: str = "cpu",
    rng=None,
):
    """Sample input sequences and targets shifted one token forward."""

    import torch

    if batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1."
        )

    if block_size < 1:
        raise ValueError(
            "block_size must be at least 1."
        )

    if len(data) <= block_size:
        raise ValueError(
            "Data must contain more tokens than block_size."
        )

    if rng is not None:
        starts = rng.integers(
            0,
            len(data) - block_size,
            size=batch_size,
        )
    else:
        starts = np.random.randint(
            0,
            len(data) - block_size,
            size=batch_size,
        )

    x = np.stack(
        [
            data[s:s + block_size]
            for s in starts
        ]
    ).astype(np.int64)

    y = np.stack(
        [
            data[s + 1:s + block_size + 1]
            for s in starts
        ]
    ).astype(np.int64)

    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(y).to(device),
    )