import numpy as np

from gpt.data import prepare_data
from gpt.tokenizer import BPETokenizer


def test_documents_receive_end_tokens(tmp_path):
    raw_dir = tmp_path / "documents"
    raw_dir.mkdir()

    documents = [
        "The first document contains some text. " * 20,
        "The second document contains different text. " * 20,
        "The third document contains more text. " * 20,
        "The fourth document contains additional text. " * 20,
    ]

    for index, text in enumerate(documents):
        path = raw_dir / f"document_{index}.txt"
        path.write_text(text, encoding="utf-8")

    output_dir = tmp_path / "processed"

    stats = prepare_data(
        raw_dir,
        output_dir,
        val_fraction=0.25,
        vocab_size=300,
        seed=42,
    )

    tokenizer = BPETokenizer.load(
        output_dir / "tokenizer.json"
    )

    train = np.fromfile(
        output_dir / "train.bin",
        dtype=np.uint16,
    )

    val = np.fromfile(
        output_dir / "val.bin",
        dtype=np.uint16,
    )

    eod_id = tokenizer.end_of_document_id

    assert stats["split_method"] == "document"
    assert stats["train_documents"] == 3
    assert stats["val_documents"] == 1

    assert np.count_nonzero(train == eod_id) == 3
    assert np.count_nonzero(val == eod_id) == 1

    assert train.max() < tokenizer.vocab_size
    assert val.max() < tokenizer.vocab_size


def test_end_token_survives_tokenizer_save_and_load(tmp_path):
    text = "A document containing enough repeated text. " * 20

    tokenizer = BPETokenizer.train(
        text,
        vocab_size=300,
    )

    original_id = tokenizer.end_of_document_id

    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)

    loaded = BPETokenizer.load(path)

    assert loaded.end_of_document_id == original_id