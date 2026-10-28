"""Tokenize the raw dataset and write training/validation binaries."""
from gpt.data import prepare_data

stats = prepare_data(
"data/raw/tinyshakespeare.txt",
"data/processed",
)

for key, value in stats.items():
    print(f"{key}: {value:,}")
