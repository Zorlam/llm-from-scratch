"""Dataset preparation and batch sampling."""
from pathlib import Path
import numpy as np
from gpt.tokenizer import CharTokenizer


def prepare_data(raw_path: str | Path, out_dir: str | Path, val_fraction: float = 0.1) -> dict:
    """Prepare the text dataset and save training and validation files."""
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1.")

    text = Path(raw_path).read_text(encoding="utf-8")
    tokenizer = CharTokenizer.from_text(text)
    token_ids = np.array(tokenizer.encode(text), dtype=np.uint16)

    split = int(len(token_ids) * (1 - val_fraction))
    train_ids = token_ids[:split]
    val_ids = token_ids[split:]

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


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: str = "cpu", rng=None):
    """Sample input sequences and targets shifted one character forward."""
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if block_size < 1:
        raise ValueError("block_size must be at least 1.")
    if len(data) <= block_size:
        raise ValueError("Data must contain more tokens than block_size.")

    rng = rng if rng is not None else np.random.default_rng()
    starts = rng.integers(0, len(data) - block_size, size=batch_size)

    x = np.stack([data[s:s + block_size] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1:s + block_size + 1] for s in starts]).astype(np.int64)

    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)