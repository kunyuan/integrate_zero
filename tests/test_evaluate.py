"""Tests for the Evaluation module (Task 13).

Tests the Evaluator class which provides aggregate evaluation of the
model on a list of integration problems, plus a SymPy baseline.
"""

import sympy
from sympy import Symbol, Integral

from integrate_zero.eval.evaluate import Evaluator
from integrate_zero.model.transformer import IntegrateZeroModel
from integrate_zero.data.vocabulary import Vocabulary

x = Symbol("x")


# ---------------------------------------------------------------------------
# test_evaluator_runs
# ---------------------------------------------------------------------------

def test_evaluator_runs():
    """Evaluator.evaluate() should return a dict with the expected keys."""
    vocab = Vocabulary()
    model = IntegrateZeroModel(
        vocab_size=len(vocab), d_model=64, nhead=2,
        num_layers=2, d_ff=128,
    )
    evaluator = Evaluator(model, vocab)
    problems = [Integral(x**2, x), Integral(sympy.sin(x), x)]
    results = evaluator.evaluate(problems, search_budget=10, num_candidates=4)
    assert "solve_rate" in results
    assert "avg_steps" in results
    assert 0.0 <= results["solve_rate"] <= 1.0


# ---------------------------------------------------------------------------
# test_evaluator_sympy_baseline
# ---------------------------------------------------------------------------

def test_evaluator_sympy_baseline():
    """SymPy should be able to solve basic integrals like x**2 and sin(x)."""
    evaluator = Evaluator(model=None, vocab=None)
    problems = [Integral(x**2, x), Integral(sympy.sin(x), x)]
    results = evaluator.sympy_baseline(problems)
    assert results["solve_rate"] == 1.0


# ---------------------------------------------------------------------------
# test_evaluator_empty_problems
# ---------------------------------------------------------------------------

def test_evaluator_empty_problems():
    """An empty problem list should produce solve_rate=0.0 and total=0."""
    evaluator = Evaluator(model=None, vocab=None)
    results = evaluator.sympy_baseline([])
    assert results["solve_rate"] == 0.0
    assert results["total"] == 0


# ---------------------------------------------------------------------------
# Additional tests
# ---------------------------------------------------------------------------

def test_evaluate_result_keys():
    """evaluate() results should contain all four required keys."""
    vocab = Vocabulary()
    model = IntegrateZeroModel(
        vocab_size=len(vocab), d_model=64, nhead=2,
        num_layers=2, d_ff=128,
    )
    evaluator = Evaluator(model, vocab)
    problems = [Integral(x**2, x)]
    results = evaluator.evaluate(problems, search_budget=10, num_candidates=4)
    assert "solve_rate" in results
    assert "avg_steps" in results
    assert "total" in results
    assert "solved" in results
    assert results["total"] == 1


def test_sympy_baseline_result_keys():
    """sympy_baseline() results should contain all three required keys."""
    evaluator = Evaluator(model=None, vocab=None)
    problems = [Integral(x**2, x)]
    results = evaluator.sympy_baseline(problems)
    assert "solve_rate" in results
    assert "total" in results
    assert "solved" in results
    assert results["total"] == 1
    assert results["solved"] == 1


def test_evaluate_empty_problems():
    """evaluate() with empty problem list should return zeros."""
    evaluator = Evaluator(model=None, vocab=None)
    results = evaluator.evaluate([], search_budget=10, num_candidates=4)
    assert results["solve_rate"] == 0.0
    assert results["total"] == 0
    assert results["solved"] == 0
    assert results["avg_steps"] == 0.0


def test_sympy_baseline_hard_problem():
    """SymPy may not solve very hard integrals -- solve_rate can be < 1.0."""
    evaluator = Evaluator(model=None, vocab=None)
    # x**2 is easy; SymPy can solve it
    # A problem that remains unevaluated would cause solve_rate < 1.0
    # We just check the structure is valid
    problems = [Integral(x**2, x)]
    results = evaluator.sympy_baseline(problems)
    assert 0.0 <= results["solve_rate"] <= 1.0
    assert results["total"] == len(problems)
