"""Train or resume a small GPT language model."""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from gpt.data import get_batch
from gpt.model import GPTConfig, GPTLanguageModel
from gpt.tokenizer import CharTokenizer


DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")
DEVICE = "cpu"

BATCH_SIZE = 16
BLOCK_SIZE = 64
EVAL_INTERVAL = 200
EVAL_BATCHES = 20
LEARNING_RATE = 3e-4

N_EMBD = 64
N_HEAD = 4
N_LAYER = 2
DROPOUT = 0.1
SEED = 1337


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train or resume the GPT language model."
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Total training step to reach.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path to load before continuing training.",
    )

    args = parser.parse_args()

    if args.max_steps < 1:
        parser.error("--max-steps must be at least 1.")

    return args


def save_checkpoint(path, model, optimizer, config, step, val_loss):
    """Save model weights, optimizer state, and training progress."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "vocab_size": config.vocab_size,
                "block_size": config.block_size,
                "n_embd": config.n_embd,
                "n_head": config.n_head,
                "n_layer": config.n_layer,
                "dropout": config.dropout,
            },
            "step": step,
            "val_loss": val_loss,
        },
        path,
    )


def main():
    args = parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print("Loading dataset...")

    train_data = np.fromfile(
        DATA_DIR / "train.bin",
        dtype=np.uint16,
    )
    val_data = np.fromfile(
        DATA_DIR / "val.bin",
        dtype=np.uint16,
    )
    tokenizer = CharTokenizer.load(
        DATA_DIR / "tokenizer.json"
    )

    print(f"Training tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    checkpoint = None

    if args.resume:
        checkpoint_path = Path(args.resume)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=True,
        )

        print(f"Loading checkpoint: {checkpoint_path}")

        config = GPTConfig(**checkpoint["config"])

        if config.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                "Checkpoint vocabulary size does not match "
                "the current tokenizer."
            )

    else:
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=BLOCK_SIZE,
            n_embd=N_EMBD,
            n_head=N_HEAD,
            n_layer=N_LAYER,
            dropout=DROPOUT,
        )

    model = GPTLanguageModel(config).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    start_step = 0
    best_val_loss = math.inf

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])

        start_step = checkpoint["step"]
        best_val_loss = checkpoint["val_loss"]

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )
            print("Optimizer state restored.")
        else:
            print(
                "This checkpoint has no optimizer state. "
                "Model weights loaded; optimizer starts fresh."
            )

        print(f"Checkpoint step: {start_step}")
        print(f"Previous validation loss: {best_val_loss:.4f}")

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(f"Model parameters: {parameter_count:,}")

    def estimate_loss():
        model.eval()
        results = {}

        with torch.no_grad():
            for name, data in (
                ("train", train_data),
                ("val", val_data),
            ):
                losses = []

                for _ in range(EVAL_BATCHES):
                    x, y = get_batch(
                        data,
                        batch_size=BATCH_SIZE,
                        block_size=config.block_size,
                        device=DEVICE,
                    )

                    _, loss = model(x, y)
                    losses.append(loss.item())

                results[name] = sum(losses) / len(losses)

        model.train()
        return results

    if start_step >= args.max_steps:
        print(
            f"Checkpoint is already at step {start_step}. "
            f"Choose --max-steps greater than {start_step} "
            "to continue training."
        )
        return

    start_time = time.time()

    print(
        f"\nTraining from step {start_step} "
        f"to step {args.max_steps}...\n"
    )

    model.train()

    for step in range(start_step + 1, args.max_steps + 1):
        x, y = get_batch(
            train_data,
            batch_size=BATCH_SIZE,
            block_size=config.block_size,
            device=DEVICE,
        )

        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        if step == 1 or step % EVAL_INTERVAL == 0 or step == args.max_steps:
            losses = estimate_loss()
            elapsed = time.time() - start_time

            print(
                f"Step {step:5d}/{args.max_steps} | "
                f"train loss: {losses['train']:.4f} | "
                f"val loss: {losses['val']:.4f} | "
                f"time: {elapsed:.1f}s"
            )

            save_checkpoint(
                CHECKPOINT_DIR / "latest_model.pt",
                model,
                optimizer,
                config,
                step,
                losses["val"],
            )

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

                save_checkpoint(
                    CHECKPOINT_DIR / "best_model.pt",
                    model,
                    optimizer,
                    config,
                    step,
                    best_val_loss,
                )

                print("  Saved new best checkpoint.")

    print("\nTraining complete.")
    print(f"Latest checkpoint: {CHECKPOINT_DIR / 'latest_model.pt'}")
    print(f"Best checkpoint: {CHECKPOINT_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    main()