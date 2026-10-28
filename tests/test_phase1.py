import numpy as np
import pytest

from gpt.data import get_batch, prepare_data
from gpt.tokenizer import CharTokenizer


def test_roundtrip():
    text = "Hello, World!\nSecond line."
    tokenizer = CharTokenizer.from_text(text)
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_save_load(tmp_path):
    tokenizer = CharTokenizer.from_text("abcabc xyz")
    tokenizer.save(tmp_path / "tokenizer.json")
    loaded = CharTokenizer.load(tmp_path / "tokenizer.json")
    assert loaded.chars == tokenizer.chars
    assert loaded.encode("abc") == tokenizer.encode("abc")


def test_prepare_data(tmp_path):
    raw = tmp_path / "raw.txt"
    raw.write_text("the quick brown fox jumps over the lazy dog " * 50)

    stats = prepare_data(raw, tmp_path / "out", val_fraction=0.1)

    assert stats["train_tokens"] + stats["val_tokens"] == stats["characters"]

    train = np.fromfile(tmp_path / "out" / "train.bin", dtype=np.uint16)
    assert len(train) == stats["train_tokens"]
    assert train.max() < stats["vocab_size"]


def test_batch_is_shifted_by_one():
    pytest.importorskip("torch")

    data = np.arange(1000, dtype=np.uint16)
    x, y = get_batch(
        data,
        batch_size=4,
        block_size=8,
        rng=np.random.default_rng(0),
    )

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert (y[:, :-1] == x[:, 1:]).all()
    assert (y[:, -1] == x[:, -1] + 1).all()