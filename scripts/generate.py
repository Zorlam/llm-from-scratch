"""Load a trained GPT checkpoint and generate text."""

import argparse
from pathlib import Path

import torch

from gpt.model import GPTConfig, GPTLanguageModel
from gpt.tokenizer import CharTokenizer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path("data/processed")
CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
DEVICE = "cpu"


# --------------------------------------------------
# Load the tokenizer and checkpoint
# --------------------------------------------------

def load_model():
    """Load the tokenizer, model configuration, and trained weights."""

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. "
            "Train the model first."
        )

    tokenizer = CharTokenizer.load(
        DATA_DIR / "tokenizer.json"
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    config = GPTConfig(**checkpoint["config"])

    model = GPTLanguageModel(config).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from training step {checkpoint['step']}.")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    return model, tokenizer


# --------------------------------------------------
# Generate text
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate text with your trained GPT model."
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
        help="Number of new characters to generate.",
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

    try:
        prompt_ids = tokenizer.encode(args.prompt)
    except KeyError as error:
        parser.error(
            f"The prompt contains a character missing from the "
            f"tokenizer vocabulary: {error}"
        )

    if not prompt_ids:
        parser.error("The prompt cannot be empty.")

    context = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=DEVICE,
    )

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
    output_path.write_text(
        generated_text,
        encoding="utf-8",
    )

    print(f"\nGenerated text saved to: {output_path}")


if __name__ == "__main__":
    main()