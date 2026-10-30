"""Character-level and Byte Pair Encoding tokenizers."""

import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers


class CharTokenizer:
    """Character-level tokenizer: every unique character is one token."""

    def __init__(self, chars: list[str]):
        self.chars = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"type": "char", "chars": self.chars}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        data = json.loads(
            Path(path).read_text(encoding="utf-8")
        )
        return cls(data["chars"])


class BPETokenizer:
    """Byte-level Byte Pair Encoding tokenizer."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = 8000,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        """Train a BPE tokenizer on a text corpus."""

        if not text:
            raise ValueError("Training text cannot be empty.")

        if vocab_size < 300:
            raise ValueError(
                "vocab_size must be at least 300 for byte-level BPE."
            )

        if min_frequency < 1:
            raise ValueError(
                "min_frequency must be at least 1."
            )

        tokenizer = Tokenizer(
            models.BPE(unk_token="[UNK]")
        )

        byte_level = pre_tokenizers.ByteLevel(
            add_prefix_space=False
        )

        tokenizer.pre_tokenizer = byte_level
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["[UNK]"],
            initial_alphabet=byte_level.alphabet(),
        )

        tokenizer.train_from_iterator(
            [text],
            trainer=trainer,
        )

        return cls(tokenizer)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def save(self, path: str | Path) -> None:
        self.tokenizer.save(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        tokenizer = Tokenizer.from_file(str(path))
        return cls(tokenizer)