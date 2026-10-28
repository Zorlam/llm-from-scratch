"""Download the Tiny Shakespeare dataset."""
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT = Path("data/raw/tinyshakespeare.txt")

OUT.parent.mkdir(parents=True, exist_ok=True)

if OUT.exists():
    print(f"Dataset already exists: {OUT} ({OUT.stat().st_size:,} bytes)")
else:
    urllib.request.urlretrieve(URL, OUT)
    print(f"Downloaded {OUT} ({OUT.stat().st_size:,} bytes)")
