"""Dataset loading, preparation, and batch sampling."""

from pathlib import Path

import numpy as np

from gpt.tokenizer import BPETokenizer


def load_documents(input_path: str | Path) -> list[tuple[Path, str]]:
    """Load text documents from a file or recursively from a directory.

    Returns:
        A list of (source_path, document_text) tuples.
    """

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


def prepare_data(
    raw_path: str | Path,
    out_dir: str | Path,
    val_fraction: float = 0.1,
    vocab_size: int = 8000,
    min_frequency: int = 2,
) -> dict:
    """Prepare a BPE-tokenized dataset."""

    if not 0 < val_fraction < 1:
        raise ValueError(
            "val_fraction must be between 0 and 1."
        )

    text = Path(raw_path).read_text(encoding="utf-8")

    if not text:
        raise ValueError("Dataset cannot be empty.")

    split = int(len(text) * (1 - val_fraction))
    train_text = text[:split]
    val_text = text[split:]

    print("Training BPE tokenizer...")
    tokenizer = BPETokenizer.train(
        train_text,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )

    train_ids = np.array(
        tokenizer.encode(train_text),
        dtype=np.uint16,
    )
    val_ids = np.array(
        tokenizer.encode(val_text),
        dtype=np.uint16,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")
    tokenizer.save(out_dir / "tokenizer.json")

    return {
        "characters": len(text),
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
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