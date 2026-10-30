import numpy as np
import pytest

from gpt.data import get_batch, prepare_data
from gpt.tokenizer import CharTokenizer, BPETokenizer


def test_char_tokenizer_roundtrip():
    text = "Hello, World!\nSecond line."
    tokenizer = CharTokenizer.from_text(text)

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_char_tokenizer_save_load(tmp_path):
    tokenizer = CharTokenizer.from_text("abcabc xyz")
    tokenizer.save(tmp_path / "char_tokenizer.json")

    loaded = CharTokenizer.load(
        tmp_path / "char_tokenizer.json"
    )

    assert loaded.chars == tokenizer.chars
    assert loaded.encode("abc") == tokenizer.encode("abc")


def test_bpe_tokenizer_roundtrip():
    text = "Hello, World!\nSecond line. " * 20

    tokenizer = BPETokenizer.train(
        text,
        vocab_size=300,
    )

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert decoded == text
    assert len(encoded) < len(text)


def test_bpe_tokenizer_save_load(tmp_path):
    text = "abcabc xyz " * 50

    tokenizer = BPETokenizer.train(
        text,
        vocab_size=300,
    )

    tokenizer.save(tmp_path / "tokenizer.json")

    loaded = BPETokenizer.load(
        tmp_path / "tokenizer.json"
    )

    assert loaded.vocab_size == tokenizer.vocab_size
    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.decode(loaded.encode(text)) == text


def test_prepare_data(tmp_path):
    text = "the quick brown fox jumps over the lazy dog " * 50

    raw = tmp_path / "raw.txt"
    raw.write_text(text, encoding="utf-8")

    output_dir = tmp_path / "out"

    stats = prepare_data(
        raw,
        output_dir,
        val_fraction=0.1,
    )

    train = np.fromfile(
        output_dir / "train.bin",
        dtype=np.uint16,
    )

    val = np.fromfile(
        output_dir / "val.bin",
        dtype=np.uint16,
    )

    tokenizer = BPETokenizer.load(
        output_dir / "tokenizer.json"
    )

    split = int(len(text) * 0.9)

    train_text = text[:split]
    val_text = text[split:]

    assert stats["characters"] == len(text)

    assert len(train) == stats["train_tokens"]
    assert len(val) == stats["val_tokens"]

    assert stats["train_tokens"] + stats["val_tokens"] < stats["characters"]

    assert train.max() < tokenizer.vocab_size
    assert val.max() < tokenizer.vocab_size

    assert tokenizer.decode(train.tolist()) == train_text
    assert tokenizer.decode(val.tolist()) == val_text


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
