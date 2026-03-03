"""Tests for the IntegrateZero transformer model with value head."""

import torch

from integrate_zero.model.transformer import IntegrateZeroModel
from integrate_zero.data.vocabulary import Vocabulary


def test_model_forward_shape():
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                               num_layers=2, d_ff=128)
    input_ids = torch.randint(0, len(vocab), (2, 20))
    logits, value = model(input_ids, sep_positions=torch.tensor([5, 5]))
    assert logits.shape == (2, 20, len(vocab))
    assert value.shape == (2,)
    assert (value >= 0).all() and (value <= 1).all()


def test_model_generate():
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                               num_layers=2, d_ff=128)
    bos = vocab.token_to_id("BOS")
    sep = vocab.token_to_id("SEP")
    sin_id = vocab.token_to_id("sin")
    x_id = vocab.token_to_id("x")
    prompt = torch.tensor([[bos, sin_id, x_id, sep]])
    generated = model.generate(prompt, max_new_tokens=10, vocab=vocab)
    assert generated.shape[0] == 1
    assert generated.shape[1] > 4


def test_model_param_count():
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=384, nhead=6,
                               num_layers=8, d_ff=1536)
    total = sum(p.numel() for p in model.parameters())
    assert 10_000_000 < total < 20_000_000


def test_model_forward_without_explicit_sep_positions():
    """When sep_positions is None, model should auto-detect SEP tokens."""
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                               num_layers=2, d_ff=128)
    bos = vocab.token_to_id("BOS")
    sep = vocab.token_to_id("SEP")
    sin_id = vocab.token_to_id("sin")
    x_id = vocab.token_to_id("x")
    input_ids = torch.tensor([[bos, sin_id, x_id, sep, x_id, x_id]])
    logits, value = model(input_ids)
    assert logits.shape == (1, 6, len(vocab))
    assert value.shape == (1,)
    assert (value >= 0).all() and (value <= 1).all()


def test_model_causal_mask():
    """Verify the model uses causal masking by checking that changing future
    tokens does not affect logits at earlier positions."""
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                               num_layers=2, d_ff=128)
    model.eval()

    bos = vocab.token_to_id("BOS")
    sep = vocab.token_to_id("SEP")
    sin_id = vocab.token_to_id("sin")
    x_id = vocab.token_to_id("x")
    add_id = vocab.token_to_id("add")

    # Two sequences that share the first 4 tokens but differ at position 4
    seq_a = torch.tensor([[bos, sin_id, x_id, sep, x_id, x_id]])
    seq_b = torch.tensor([[bos, sin_id, x_id, sep, add_id, x_id]])

    with torch.no_grad():
        logits_a, _ = model(seq_a, sep_positions=torch.tensor([3]))
        logits_b, _ = model(seq_b, sep_positions=torch.tensor([3]))

    # Logits at positions 0-3 should be identical since future tokens differ
    assert torch.allclose(logits_a[0, :4], logits_b[0, :4], atol=1e-5)
    # Logits at position 4 should differ (different input at that position)
    assert not torch.allclose(logits_a[0, 4], logits_b[0, 4], atol=1e-5)


def test_model_padding_does_not_affect_output():
    """Padding tokens should not affect logits of non-padding positions."""
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                               num_layers=2, d_ff=128)
    model.eval()

    bos = vocab.token_to_id("BOS")
    sep = vocab.token_to_id("SEP")
    sin_id = vocab.token_to_id("sin")
    x_id = vocab.token_to_id("x")
    pad = vocab.pad_id

    # Same content, padded vs unpadded
    seq_short = torch.tensor([[bos, sin_id, x_id, sep, x_id]])
    seq_padded = torch.tensor([[bos, sin_id, x_id, sep, x_id, pad, pad]])

    with torch.no_grad():
        logits_short, val_short = model(seq_short, sep_positions=torch.tensor([3]))
        logits_padded, val_padded = model(seq_padded, sep_positions=torch.tensor([3]))

    # Logits for the non-padded prefix should be very close
    assert torch.allclose(logits_short[0, :5], logits_padded[0, :5], atol=1e-5)
    # Value heads should agree
    assert torch.allclose(val_short, val_padded, atol=1e-5)
