
"""Load a trained BPE GPT checkpoint and generate text."""

import argparse
import hashlib
from pathlib import Path

import torch

from gpt.model import GPTConfig, GPTLanguageModel
from gpt.tokenizer import BPETokenizer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path("data/processed_bpe")
CHECKPOINT_PATH = Path("checkpoints/bpe/best_model.pt")
DEVICE = "cpu"


# --------------------------------------------------
# Load the tokenizer and checkpoint
# --------------------------------------------------

def get_tokenizer_fingerprint(path):
    """Create a SHA-256 fingerprint of the tokenizer file."""

    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def load_model():
    """Load the tokenizer, model configuration, and trained weights."""

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. "
            "Train the BPE model first."
        )

    tokenizer_path = DATA_DIR / "tokenizer.json"

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {tokenizer_path}. "
            "Prepare the BPE dataset first."
        )

    tokenizer = BPETokenizer.load(tokenizer_path)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    if "tokenizer_fingerprint" not in checkpoint:
        raise ValueError(
            "Checkpoint does not contain a tokenizer fingerprint. "
            "It cannot be safely used with the BPE generation pipeline."
        )

    tokenizer_fingerprint = get_tokenizer_fingerprint(
        tokenizer_path
    )

    if (
        checkpoint["tokenizer_fingerprint"]
        != tokenizer_fingerprint
    ):
        raise ValueError(
            "Checkpoint tokenizer does not match the current BPE tokenizer."
        )

    config = GPTConfig(**checkpoint["config"])

    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            "Checkpoint vocabulary size does not match "
            "the current tokenizer."
        )

    model = GPTLanguageModel(config).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Loaded checkpoint from training step "
        f"{checkpoint['step']}."
    )

    print(
        f"Validation loss: {checkpoint['val_loss']:.4f}"
    )

    print(
        f"Vocabulary size: {tokenizer.vocab_size:,}"
    )

    return model, tokenizer


# --------------------------------------------------
# Generate text
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate text with your trained BPE GPT model."
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="\n",
        help="Starting text for generation.",
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=500,
        help="Number of new tokens to generate.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Controls randomness. Lower values are more predictable.",
    )

    args = parser.parse_args()

    if args.tokens < 0:
        parser.error("--tokens cannot be negative.")

    if args.temperature <= 0:
        parser.error("--temperature must be greater than zero.")

    model, tokenizer = load_model()

    prompt_ids = tokenizer.encode(args.prompt)

    if not prompt_ids:
        parser.error("The prompt cannot be empty.")

    if len(prompt_ids) > model.config.block_size:
        parser.error(
            f"The prompt contains {len(prompt_ids)} tokens, "
            f"but the model supports a maximum context of "
            f"{model.config.block_size} tokens."
        )

    context = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():
        generated = model.generate(
            context,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
        )

    generated_text = tokenizer.decode(
        generated[0].tolist()
    )

    print("\n" + "=" * 50)
    print("GENERATED TEXT")
    print("=" * 50)
    print(generated_text)

    output_path = Path("checkpoints/generated_sample.txt")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        generated_text,
        encoding="utf-8",
    )

    print(f"\nGenerated text saved to: {output_path}")


if __name__ == "__main__":
    main()
