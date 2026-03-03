"""Tests for MCTS-Model integration (Task 11).

Tests that _generate_candidates and _evaluate correctly bridge the
neural network model with the MCTS search tree.
"""

import torch
import sympy
from sympy import Symbol, Integral

from integrate_zero.mcts.search import MCTS, MCTSNode
from integrate_zero.model.transformer import IntegrateZeroModel
from integrate_zero.data.vocabulary import Vocabulary

x = Symbol("x")


# ---------------------------------------------------------------------------
# _generate_candidates
# ---------------------------------------------------------------------------

def test_mcts_generates_candidates():
    """_generate_candidates should return a list of (expr, prior) tuples."""
    vocab = Vocabulary()
    model = IntegrateZeroModel(
        vocab_size=len(vocab), d_model=64, nhead=2,
        num_layers=2, d_ff=128,
    )
    mcts = MCTS(model=model, vocab=vocab, num_candidates=4, search_budget=20)
    problem = Integral(x**2, x)
    candidates = mcts._generate_candidates(problem)
    assert isinstance(candidates, list)
    # With a random model, some candidates may fail to parse -- that's OK
    for expr, prior in candidates:
        assert isinstance(expr, sympy.Basic)
        assert isinstance(prior, float)


def test_mcts_generates_candidates_returns_valid_priors():
    """Each prior should be a non-negative float."""
    vocab = Vocabulary()
    model = IntegrateZeroModel(
        vocab_size=len(vocab), d_model=64, nhead=2,
        num_layers=2, d_ff=128,
    )
    mcts = MCTS(model=model, vocab=vocab, num_candidates=4)
    problem = Integral(sympy.sin(x), x)
    candidates = mcts._generate_candidates(problem)
    for _, prior in candidates:
        assert prior >= 0.0


def test_mcts_generates_candidates_graceful_on_bad_parse():
    """_generate_candidates should not raise even if model output is gibberish."""
    vocab = Vocabulary()
    model = IntegrateZeroModel(
        vocab_size=len(vocab), d_model=64, nhead=2,
        num_layers=2, d_ff=128,
    )
    # With a random (untrained) model, most sequences will be gibberish.
    # The method should handle this gracefully by returning fewer candidates.
    mcts = MCTS(model=model, vocab=vocab, num_candidates=8)
    problem = Integral(x, x)
    candidates = mcts._generate_candidates(problem)
    assert isinstance(candidates, list)
    # Could be empty or partial -- no crash is the key assertion.


# ---------------------------------------------------------------------------
# _evaluate
# ---------------------------------------------------------------------------

def test_mcts_evaluate_terminal():
    """A terminal node (no Integral) should evaluate to 1.0."""
    vocab = Vocabulary()
    model = IntegrateZeroModel(
        vocab_size=len(vocab), d_model=64, nhead=2,
        num_layers=2, d_ff=128,
    )
    mcts = MCTS(model=model, vocab=vocab)
    node = MCTSNode(state=x**2, prior=1.0)  # terminal (no Integral)
    val = mcts._evaluate(node)
    assert val == 1.0


def test_mcts_evaluate_nonterminal():
    """A non-terminal node should return a float in [0, 1] from the value head."""
    vocab = Vocabulary()
    model = IntegrateZeroModel(
        vocab_size=len(vocab), d_model=64, nhead=2,
        num_layers=2, d_ff=128,
    )
    mcts = MCTS(model=model, vocab=vocab)
    node = MCTSNode(state=Integral(x**2, x), prior=1.0)
    val = mcts._evaluate(node)
    assert isinstance(val, float)
    assert 0.0 <= val <= 1.0


def test_mcts_evaluate_uses_model_not_placeholder():
    """After Task 11 integration, _evaluate should not return the old 0.5 placeholder
    for all non-terminal nodes (with high probability, a random model won't output
    exactly 0.5)."""
    vocab = Vocabulary()
    model = IntegrateZeroModel(
        vocab_size=len(vocab), d_model=64, nhead=2,
        num_layers=2, d_ff=128,
    )
    mcts = MCTS(model=model, vocab=vocab)
    # Evaluate several different expressions; at least one should differ from 0.5
    exprs = [
        Integral(x**2, x),
        Integral(sympy.sin(x), x),
        Integral(sympy.exp(x), x),
    ]
    values = []
    for expr in exprs:
        node = MCTSNode(state=expr, prior=1.0)
        values.append(mcts._evaluate(node))
    # It's astronomically unlikely that a random model produces exactly 0.5
    # for all three different expressions.
    assert not all(v == 0.5 for v in values)


def test_mcts_evaluate_no_grad():
    """_evaluate should not accumulate gradients on the model parameters."""
    vocab = Vocabulary()
    model = IntegrateZeroModel(
        vocab_size=len(vocab), d_model=64, nhead=2,
        num_layers=2, d_ff=128,
    )
    mcts = MCTS(model=model, vocab=vocab)
    node = MCTSNode(state=Integral(x**2, x), prior=1.0)

    # Clear any existing grads
    model.zero_grad()
    _ = mcts._evaluate(node)
    # No parameter should have a gradient
    for param in model.parameters():
        assert param.grad is None


# ---------------------------------------------------------------------------
# Fallback: model=None should still work (backward compat with placeholder)
# ---------------------------------------------------------------------------

def test_mcts_generate_candidates_no_model():
    """If model is None, _generate_candidates returns empty list."""
    mcts = MCTS(model=None, vocab=None)
    result = mcts._generate_candidates(Integral(x, x))
    assert result == []


def test_mcts_evaluate_no_model():
    """If model is None, _evaluate returns 0.5 (placeholder fallback)."""
    mcts = MCTS(model=None, vocab=None)
    node = MCTSNode(state=Integral(x, x), prior=1.0)
    assert mcts._evaluate(node) == 0.5
