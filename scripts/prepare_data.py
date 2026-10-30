"""Prepare a BPE-tokenized training dataset."""

import argparse

from gpt.data import prepare_data


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a BPE-tokenized language modelling dataset."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/tinyshakespeare.txt",
        help="Path to the raw text dataset.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/processed_bpe",
        help="Directory for processed data.",
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8000,
        help="Maximum BPE vocabulary size.",
    )

    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum frequency required for a merge.",
    )

    args = parser.parse_args()

    stats = prepare_data(
        args.input,
        args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    print("\nDataset preparation complete.")

    for key, value in stats.items():
        print(f"{key}: {value:,}")


if __name__ == "__main__":
    main()
