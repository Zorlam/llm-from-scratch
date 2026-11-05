"""Dataset loading, preparation, and batch sampling."""

import hashlib
from pathlib import Path

import numpy as np

from gpt.tokenizer import BPETokenizer


def load_documents(input_path: str | Path) -> list[tuple[Path, str]]:
    """Load text documents from a file or recursively from a directory."""

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input path does not exist: {input_path}"
        )

    if input_path.is_file():
        files = [input_path]

    elif input_path.is_dir():
        files = sorted(input_path.rglob("*.txt"))

    else:
        raise ValueError(
            f"Input path must be a file or directory: {input_path}"
        )

    documents = []

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")

        if not text.strip():
            continue

        documents.append((file_path, text))

    if not documents:
        raise ValueError(
            f"No non-empty text documents found in: {input_path}"
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
    """Prepare a deduplicated BPE-tokenized dataset."""

    if not 0 < val_fraction < 1:
        raise ValueError(
            "val_fraction must be between 0 and 1."
        )

    documents = load_documents(raw_path)

    original_document_count = len(documents)

    documents, duplicates_removed = deduplicate_documents(
        documents
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