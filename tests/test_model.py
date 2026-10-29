import torch

from gpt.model import GPTConfig, GPTLanguageModel


def test_model_forward_pass():
    config = GPTConfig(
        vocab_size=65,
        block_size=16,
        n_embd=32,
        n_head=4,
        n_layer=2,
        dropout=0.0,
    )

    model = GPTLanguageModel(config)

    x = torch.randint(0, 65, (2, 16))
    targets = torch.randint(0, 65, (2, 16))

    logits, loss = model(x, targets)

    assert logits.shape == (2, 16, 65)
    assert loss is not None
    assert torch.isfinite(loss)


def test_model_can_generate():
    config = GPTConfig(
        vocab_size=65,
        block_size=16,
        n_embd=32,
        n_head=4,
        n_layer=2,
        dropout=0.0,
    )

    model = GPTLanguageModel(config)

    starting_tokens = torch.randint(0, 65, (1, 5))
    generated = model.generate(starting_tokens, max_new_tokens=5)

    assert generated.shape == (1, 10)