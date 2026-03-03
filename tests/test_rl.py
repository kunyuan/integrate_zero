"""Tests for the RL training loop (Phase 2)."""

import torch
import sympy
from integrate_zero.train.rl import RLTrainer
from integrate_zero.model.transformer import IntegrateZeroModel
from integrate_zero.data.vocabulary import Vocabulary

x = sympy.Symbol("x")


def test_rl_collect_trajectory():
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                               num_layers=2, d_ff=128)
    trainer = RLTrainer(model, vocab, num_candidates=4, search_budget=10)
    problem = sympy.Integral(x**2, x)
    result = trainer.collect_episode(problem)
    assert "trajectory" in result
    assert "solved" in result
    assert isinstance(result["solved"], bool)


def test_rl_train_step_runs():
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                               num_layers=2, d_ff=128)
    trainer = RLTrainer(model, vocab, num_candidates=4, search_budget=10, lr=1e-3)
    problems = [sympy.Integral(sympy.diff(x**i, x), x) for i in range(2, 6)]
    stats = trainer.train_iteration(problems)
    assert "policy_loss" in stats
    assert "value_loss" in stats
    assert "solve_rate" in stats


def test_rl_train_pair_returns_losses():
    """_train_on_pair should return a dict with policy_loss and value_loss."""
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                               num_layers=2, d_ff=128)
    trainer = RLTrainer(model, vocab, num_candidates=4, search_budget=10, lr=1e-3)
    # Use simple expressions that we know convert to prefix
    state = sympy.Integral(x**2, x)
    target = x**3 / 3
    result = trainer._train_on_pair(state, target, reward=1.0)
    assert "policy_loss" in result
    assert "value_loss" in result
    assert isinstance(result["policy_loss"], float)
    assert isinstance(result["value_loss"], float)
    assert result["policy_loss"] >= 0.0
    assert result["value_loss"] >= 0.0


def test_rl_solve_rate_is_float():
    """solve_rate should be a float between 0 and 1."""
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                               num_layers=2, d_ff=128)
    trainer = RLTrainer(model, vocab, num_candidates=4, search_budget=10, lr=1e-3)
    problems = [sympy.Integral(x**2, x)]
    stats = trainer.train_iteration(problems)
    assert 0.0 <= stats["solve_rate"] <= 1.0


def test_rl_collect_episode_returns_problem():
    """collect_episode result should include the original problem."""
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                               num_layers=2, d_ff=128)
    trainer = RLTrainer(model, vocab, num_candidates=4, search_budget=10)
    problem = sympy.Integral(x**2, x)
    result = trainer.collect_episode(problem)
    assert "problem" in result
