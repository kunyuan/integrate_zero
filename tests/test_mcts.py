"""Tests for MCTS search module."""

import math

import sympy
from sympy import Symbol, Integral

from integrate_zero.mcts.search import MCTSNode, MCTS

x = Symbol("x")


# ---------------------------------------------------------------------------
# MCTSNode creation
# ---------------------------------------------------------------------------

def test_mcts_node_creation():
    expr = Integral(sympy.sin(x), x)
    node = MCTSNode(state=expr, prior=1.0)
    assert node.visit_count == 0
    assert node.value_sum == 0.0
    assert len(node.children) == 0


def test_mcts_node_defaults():
    node = MCTSNode(state=None, prior=0.7)
    assert node.parent is None
    assert node.step_type is None
    assert node.q_value == 0.0


# ---------------------------------------------------------------------------
# q_value property
# ---------------------------------------------------------------------------

def test_q_value_unvisited():
    node = MCTSNode(state=None, prior=1.0)
    assert node.q_value == 0.0


def test_q_value_visited():
    node = MCTSNode(state=None, prior=1.0)
    node.visit_count = 4
    node.value_sum = 3.0
    assert node.q_value == 0.75


# ---------------------------------------------------------------------------
# UCB score
# ---------------------------------------------------------------------------

def test_mcts_node_ucb():
    parent = MCTSNode(state=None, prior=1.0)
    parent.visit_count = 10
    child = MCTSNode(state=None, prior=0.5, parent=parent)
    child.visit_count = 2
    child.value_sum = 1.0
    ucb = child.ucb_score(c_puct=1.5)
    assert isinstance(ucb, float)
    assert ucb > 0


def test_ucb_formula():
    """Verify UCB matches the exact formula:
    q_value + c_puct * prior * sqrt(N_parent) / (1 + N_child)
    """
    parent = MCTSNode(state=None, prior=1.0)
    parent.visit_count = 16
    child = MCTSNode(state=None, prior=0.3, parent=parent)
    child.visit_count = 3
    child.value_sum = 1.5

    c_puct = 2.0
    expected_q = 1.5 / 3.0
    expected_explore = c_puct * 0.3 * math.sqrt(16) / (1 + 3)
    expected = expected_q + expected_explore

    assert abs(child.ucb_score(c_puct=c_puct) - expected) < 1e-10


def test_ucb_unvisited_node():
    """An unvisited child should have a pure exploration bonus (q=0)."""
    parent = MCTSNode(state=None, prior=1.0)
    parent.visit_count = 10
    child = MCTSNode(state=None, prior=0.8, parent=parent)
    child.visit_count = 0
    child.value_sum = 0.0

    c_puct = 1.5
    expected = 0.0 + c_puct * 0.8 * math.sqrt(10) / (1 + 0)
    assert abs(child.ucb_score(c_puct=c_puct) - expected) < 1e-10


# ---------------------------------------------------------------------------
# expand_with
# ---------------------------------------------------------------------------

def test_mcts_node_expand_valid():
    problem = Integral(sympy.sin(x), x)
    root = MCTSNode(state=problem, prior=1.0)
    candidate = -sympy.cos(x)
    child = root.expand_with(candidate, prior=0.5)
    assert child is not None
    assert len(root.children) == 1


def test_mcts_node_expand_invalid():
    problem = Integral(sympy.sin(x), x)
    root = MCTSNode(state=problem, prior=1.0)
    candidate = x**2  # wrong answer
    child = root.expand_with(candidate, prior=0.5)
    assert child is None
    assert len(root.children) == 0


def test_expand_with_stores_step_type():
    """expand_with should set step_type on the created child."""
    problem = Integral(sympy.sin(x), x)
    root = MCTSNode(state=problem, prior=1.0)
    candidate = -sympy.cos(x)
    child = root.expand_with(candidate, prior=0.5)
    assert child is not None
    assert child.step_type is not None
    # Should be INTEGRATION since we went from Integral(...) to closed form
    from integrate_zero.verify.verifier import StepType
    assert child.step_type in (StepType.INTEGRATION, StepType.IDENTITY)


def test_expand_with_sets_parent():
    problem = Integral(sympy.sin(x), x)
    root = MCTSNode(state=problem, prior=1.0)
    candidate = -sympy.cos(x)
    child = root.expand_with(candidate, prior=0.5)
    assert child.parent is root


def test_expand_with_sets_prior():
    problem = Integral(sympy.sin(x), x)
    root = MCTSNode(state=problem, prior=1.0)
    candidate = -sympy.cos(x)
    child = root.expand_with(candidate, prior=0.7)
    assert child.prior == 0.7


# ---------------------------------------------------------------------------
# backpropagate
# ---------------------------------------------------------------------------

def test_mcts_backpropagate():
    root = MCTSNode(state=None, prior=1.0)
    child = MCTSNode(state=None, prior=0.5, parent=root)
    root.children.append(child)
    child.backpropagate(1.0)
    assert child.visit_count == 1
    assert child.value_sum == 1.0
    assert root.visit_count == 1
    assert root.value_sum == 1.0


def test_backpropagate_multi_level():
    """Backpropagation should go all the way up to the root."""
    root = MCTSNode(state=None, prior=1.0)
    mid = MCTSNode(state=None, prior=0.5, parent=root)
    root.children.append(mid)
    leaf = MCTSNode(state=None, prior=0.3, parent=mid)
    mid.children.append(leaf)

    leaf.backpropagate(0.8)
    assert leaf.visit_count == 1 and leaf.value_sum == 0.8
    assert mid.visit_count == 1 and mid.value_sum == 0.8
    assert root.visit_count == 1 and root.value_sum == 0.8


def test_backpropagate_accumulates():
    """Multiple backpropagations should accumulate."""
    root = MCTSNode(state=None, prior=1.0)
    child = MCTSNode(state=None, prior=0.5, parent=root)
    root.children.append(child)

    child.backpropagate(1.0)
    child.backpropagate(0.5)
    assert child.visit_count == 2
    assert child.value_sum == 1.5
    assert root.visit_count == 2
    assert root.value_sum == 1.5


# ---------------------------------------------------------------------------
# best_child
# ---------------------------------------------------------------------------

def test_best_child_selects_highest_ucb():
    parent = MCTSNode(state=None, prior=1.0)
    parent.visit_count = 10

    c1 = MCTSNode(state=None, prior=0.8, parent=parent)
    c1.visit_count = 5
    c1.value_sum = 4.0
    parent.children.append(c1)

    c2 = MCTSNode(state=None, prior=0.2, parent=parent)
    c2.visit_count = 1
    c2.value_sum = 0.1
    parent.children.append(c2)

    best = parent.best_child(c_puct=1.5)
    # c1 should have a higher UCB because of its high q_value
    assert best is c1


# ---------------------------------------------------------------------------
# is_terminal
# ---------------------------------------------------------------------------

def test_is_terminal_with_integral():
    expr = Integral(sympy.sin(x), x)
    node = MCTSNode(state=expr, prior=1.0)
    assert not node.is_terminal()


def test_is_terminal_without_integral():
    expr = -sympy.cos(x)
    node = MCTSNode(state=expr, prior=1.0)
    assert node.is_terminal()


# ---------------------------------------------------------------------------
# MCTS class
# ---------------------------------------------------------------------------

def test_mcts_creation():
    mcts = MCTS(model=None, max_steps=10, num_candidates=4,
                search_budget=100, c_puct=2.0)
    assert mcts.max_steps == 10
    assert mcts.num_candidates == 4
    assert mcts.search_budget == 100
    assert mcts.c_puct == 2.0


def test_mcts_search_no_candidates():
    """With placeholder _generate_candidates returning empty, search returns None."""
    mcts = MCTS(model=None)
    problem = Integral(sympy.sin(x), x)
    result = mcts.search(problem)
    assert result is None


def test_mcts_select_returns_leaf():
    """_select should navigate to a leaf node (one with no children)."""
    mcts = MCTS(model=None)
    root = MCTSNode(state=None, prior=1.0)
    root.visit_count = 5

    child = MCTSNode(state=None, prior=0.5, parent=root)
    child.visit_count = 2
    child.value_sum = 1.0
    root.children.append(child)

    leaf = mcts._select(root)
    # child has no children, so it should be the selected leaf
    assert leaf is child


def test_mcts_evaluate_placeholder():
    """Placeholder _evaluate should return 0.5."""
    mcts = MCTS(model=None)
    node = MCTSNode(state=None, prior=1.0)
    assert mcts._evaluate(node) == 0.5


def test_mcts_generate_candidates_placeholder():
    """Placeholder _generate_candidates should return empty list."""
    mcts = MCTS(model=None)
    result = mcts._generate_candidates(Integral(sympy.sin(x), x))
    assert result == []
