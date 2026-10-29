"""Train a small GPT language model on the prepared text dataset."""

import math
import time
from pathlib import Path

import numpy as np
import torch

from gpt.data import get_batch
from gpt.model import GPTConfig, GPTLanguageModel
from gpt.tokenizer import CharTokenizer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = Path("data/processed")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 16
BLOCK_SIZE = 64
MAX_STEPS = 2000
EVAL_INTERVAL = 200
EVAL_BATCHES = 20
LEARNING_RATE = 3e-4

N_EMBD = 64
N_HEAD = 4
N_LAYER = 2
DROPOUT = 0.1

DEVICE = "cpu"
SEED = 1337


# --------------------------------------------------
# Reproducibility and data loading
# --------------------------------------------------

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Device: {DEVICE}")
print("Loading dataset...")

train_data = np.fromfile(DATA_DIR / "train.bin", dtype=np.uint16)
val_data = np.fromfile(DATA_DIR / "val.bin", dtype=np.uint16)
tokenizer = CharTokenizer.load(DATA_DIR / "tokenizer.json")

print(f"Training tokens: {len(train_data):,}")
print(f"Validation tokens: {len(val_data):,}")
print(f"Vocabulary size: {tokenizer.vocab_size}")


# --------------------------------------------------
# Model and optimizer
# --------------------------------------------------

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

parameter_count = sum(
    parameter.numel() for parameter in model.parameters()
)

print(f"Model parameters: {parameter_count:,}")


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def estimate_loss():
    """Estimate average training and validation loss."""
    model.eval()
    results = {}

    with torch.no_grad():
        for split_name, data in (
            ("train", train_data),
            ("val", val_data),
        ):
            losses = []

            for _ in range(EVAL_BATCHES):
                x, y = get_batch(
                    data,
                    batch_size=BATCH_SIZE,
                    block_size=BLOCK_SIZE,
                    device=DEVICE,
                )

                _, loss = model(x, y)
                losses.append(loss.item())

            results[split_name] = sum(losses) / len(losses)

    model.train()
    return results


# --------------------------------------------------
# Training loop
# --------------------------------------------------

best_val_loss = math.inf
start_time = time.time()

print("\nStarting training...\n")

for step in range(1, MAX_STEPS + 1):
    model.train()

    x, y = get_batch(
        train_data,
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
        device=DEVICE,
    )

    _, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    if step == 1 or step % EVAL_INTERVAL == 0:
        losses = estimate_loss()
        elapsed = time.time() - start_time

        print(
            f"Step {step:4d}/{MAX_STEPS} | "
            f"train loss: {losses['train']:.4f} | "
            f"val loss: {losses['val']:.4f} | "
            f"time: {elapsed:.1f}s"
        )

        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "vocab_size": config.vocab_size,
                        "block_size": config.block_size,
                        "n_embd": config.n_embd,
                        "n_head": config.n_head,
                        "n_layer": config.n_layer,
                        "dropout": config.dropout,
                    },
                    "step": step,
                    "val_loss": best_val_loss,
                },
                CHECKPOINT_DIR / "best_model.pt",
            )

            print("  Saved new best checkpoint.")

print("\nTraining complete.")

# --------------------------------------------------
# Generate a sample
# --------------------------------------------------

model.eval()

start_text = "\n"
start_ids = tokenizer.encode(start_text)
context = torch.tensor([start_ids], dtype=torch.long, device=DEVICE)

generated = model.generate(
    context,
    max_new_tokens=500,
    temperature=0.8,
)

generated_text = tokenizer.decode(generated[0].tolist())

print("\nGenerated text:\n")
print(generated_text)

output_path = CHECKPOINT_DIR / "generated_sample.txt"
output_path.write_text(generated_text, encoding="utf-8")

print(f"\nGenerated sample saved to: {output_path}")