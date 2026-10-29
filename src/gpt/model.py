"""A small GPT-style language model built with PyTorch."""

import torch
import torch.nn as nn
from torch.nn import functional as F


class GPTConfig:
    """Configuration settings for the GPT model."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 128,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        dropout: float = 0.1,
    ):
        if vocab_size < 1:
            raise ValueError("vocab_size must be positive.")
        if block_size < 1:
            raise ValueError("block_size must be positive.")
        if n_embd < 1 or n_head < 1 or n_layer < 1:
            raise ValueError("Model dimensions must be positive.")
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the range [0, 1).")

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.dropout = dropout


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention that cannot access future tokens."""

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head

        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(
            torch.ones(config.block_size, config.block_size, dtype=torch.bool)
        )
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape

        key = self.key(x)
        query = self.query(x)
        value = self.value(x)

        key = key.view(
            batch_size, seq_len, self.n_head, self.head_size
        ).transpose(1, 2)

        query = query.view(
            batch_size, seq_len, self.n_head, self.head_size
        ).transpose(1, 2)

        value = value.view(
            batch_size, seq_len, self.n_head, self.head_size
        ).transpose(1, 2)

        scores = query @ key.transpose(-2, -1)
        scores = scores * (self.head_size ** -0.5)

        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        output = weights @ value
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, channels)

        output = self.resid_dropout(self.proj(output))

        return output


class FeedForward(nn.Module):
    """Feed-forward network applied to each token representation."""

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """One Transformer block with attention and feed-forward layers."""

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.feed_forward = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))

        return x


class GPTLanguageModel(nn.Module):
    """GPT-style autoregressive language model."""

    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.n_embd
        )
        self.position_embedding = nn.Embedding(
            config.block_size, config.n_embd
        )

        self.embedding_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(config)
                for _ in range(config.n_layer)
            ]
        )

        self.final_layer_norm = nn.LayerNorm(config.n_embd)

        self.language_model_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=False
        )

        self.apply(self._initialize_weights)

        # Share the token embedding weights with the output projection.
        self.language_model_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.shape

        if seq_len > self.config.block_size:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"block_size {self.config.block_size}."
            )

        positions = torch.arange(seq_len, device=idx.device)

        token_embeddings = self.token_embedding(idx)
        position_embeddings = self.position_embedding(positions)

        x = self.embedding_dropout(
            token_embeddings + position_embeddings
        )

        x = self.blocks(x)
        x = self.final_layer_norm(x)
        logits = self.language_model_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate tokens autoregressively from a starting sequence."""

        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")

        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative.")

        was_training = self.training
        self.eval()

        try:
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size:]

                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature

                probabilities = F.softmax(logits, dim=-1)

                next_token = torch.multinomial(
                    probabilities, num_samples=1
                )

                idx = torch.cat((idx, next_token), dim=1)

        finally:
            self.train(was_training)

        return idx