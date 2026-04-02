"""Random symbolic expression generator for training data.

Builds random expression trees that:
 - contain the variable ``x``
 - consist only of operators/functions in the IntegrateZero vocabulary
 - roundtrip cleanly through prefix notation

Public API
----------
generate_expression(max_depth)       -> sympy.Expr
generate_training_pair(max_depth)    -> (f, F)  where f = dF/dx
"""

from __future__ import annotations

import random
from typing import List, Tuple

import sympy
from sympy import Symbol, Integer

from integrate_zero.data.prefix import sympy_to_prefix, prefix_to_sympy

# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------

x = Symbol("x")

_REAL_PARAMS: List[Symbol] = [
    Symbol("a", real=True),
    Symbol("b", real=True),
    Symbol("c", real=True),
    Symbol("d", real=True),
]

_INTEGER_PARAMS: List[Symbol] = [
    Symbol("k", integer=True),
    Symbol("l", integer=True),
    Symbol("m", integer=True),
    Symbol("n", integer=True),
]

_PARAMS: List[Symbol] = _REAL_PARAMS + _INTEGER_PARAMS

# Safe integer constants for leaf nodes (avoiding 0 to prevent division-by-zero
# and log(0) issues when used as arguments).
_SAFE_INTS: List[int] = [i for i in range(-10, 11) if i != 0]

# ---------------------------------------------------------------------------
# Operator / function tables
# ---------------------------------------------------------------------------

# Binary operators that combine two sub-expressions.
_BINARY_OPS = [
    lambda a, b: a + b,
    lambda a, b: a - b,
    lambda a, b: a * b,
]

# Unary functions that wrap a sub-expression.
# We only include functions that are safe for arbitrary real inputs.
_SAFE_UNARY = [
    sympy.sin,
    sympy.cos,
    sympy.exp,
    sympy.sinh,
    sympy.cosh,
    sympy.tanh,
    sympy.atan,   # arctan — defined on all reals
]

# Unary functions that need a positive argument (we wrap with x**2 + 1
# or similar to guarantee positivity).  Handled separately.
_POS_UNARY = [
    sympy.log,
    sympy.sqrt,
]


# ---------------------------------------------------------------------------
# Expression tree builder
# ---------------------------------------------------------------------------


def _random_leaf(must_contain_x: bool) -> sympy.Expr:
    """Return a random leaf node.

    Parameters
    ----------
    must_contain_x : bool
        If True, always returns ``x``.  Otherwise picks randomly among
        ``x``, integer constants, and parameter symbols.
    """
    if must_contain_x:
        return x

    choice = random.random()
    if choice < 0.50:
        return x
    elif choice < 0.75:
        return Integer(random.choice(_SAFE_INTS))
    else:
        return random.choice(_PARAMS)


def _build_tree(depth: int, max_depth: int, must_contain_x: bool) -> sympy.Expr:
    """Recursively build a random expression tree.

    Parameters
    ----------
    depth : int
        Current depth (0 = root).
    max_depth : int
        Maximum depth allowed.
    must_contain_x : bool
        If True, at least one path in this subtree must contain ``x``.
    """
    # Base case: at max depth, return a leaf.
    if depth >= max_depth:
        return _random_leaf(must_contain_x)

    # With some probability, stop early and return a leaf even before
    # max_depth (makes shallow expressions possible).
    if depth > 0 and random.random() < 0.20:
        return _random_leaf(must_contain_x)

    # Choose between binary op, safe unary, and positive-argument unary.
    roll = random.random()

    if roll < 0.50:
        # --- Binary operator ---
        op = random.choice(_BINARY_OPS)
        if must_contain_x:
            # Ensure at least one child contains x.
            # Randomly pick which child *must* have x; the other may or may not.
            if random.random() < 0.5:
                left = _build_tree(depth + 1, max_depth, must_contain_x=True)
                right = _build_tree(depth + 1, max_depth, must_contain_x=False)
            else:
                left = _build_tree(depth + 1, max_depth, must_contain_x=False)
                right = _build_tree(depth + 1, max_depth, must_contain_x=True)
        else:
            left = _build_tree(depth + 1, max_depth, must_contain_x=False)
            right = _build_tree(depth + 1, max_depth, must_contain_x=False)
        return op(left, right)

    elif roll < 0.85:
        # --- Safe unary function ---
        fn = random.choice(_SAFE_UNARY)
        child = _build_tree(depth + 1, max_depth, must_contain_x=must_contain_x)
        return fn(child)

    else:
        # --- Positive-argument unary (log, sqrt) ---
        fn = random.choice(_POS_UNARY)
        # Build inner expression, then square it and add 1 to ensure positivity.
        inner = _build_tree(depth + 1, max_depth, must_contain_x=must_contain_x)
        safe_arg = inner ** 2 + 1
        return fn(safe_arg)


def _is_prefix_safe(expr: sympy.Expr) -> bool:
    """Check whether *expr* roundtrips cleanly through prefix notation."""
    try:
        tokens = sympy_to_prefix(expr)
        _ = prefix_to_sympy(tokens)
        return True
    except (ValueError, TypeError, AttributeError, AssertionError):
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_expression(max_depth: int = 3, *, _max_attempts: int = 200) -> sympy.Expr:
    """Generate a random symbolic expression containing ``x``.

    Parameters
    ----------
    max_depth : int
        Controls tree depth (typically 2-8).

    Returns
    -------
    sympy.Expr
        A random expression that contains ``x`` and roundtrips through
        prefix notation.
    """
    for _ in range(_max_attempts):
        expr = _build_tree(depth=0, max_depth=max_depth, must_contain_x=True)

        # SymPy may simplify the expression in a way that removes x or
        # produces constructs outside the vocabulary.  Validate.
        if x not in expr.free_symbols:
            continue
        if not _is_prefix_safe(expr):
            continue
        return expr

    # Fallback: guaranteed to work.
    return x ** 2 + x


def generate_training_pair(
    max_depth: int = 3, *, seed: int | None = None, _max_attempts: int = 100
) -> Tuple[sympy.Expr, sympy.Expr]:
    """Generate a training pair ``(f, F)`` where ``f = dF/dx``.

    The antiderivative ``F`` is generated randomly, then ``f = diff(F, x)``
    is computed symbolically.  Pairs where ``f == 0`` or ``x`` is not in
    ``f.free_symbols`` are rejected.

    Parameters
    ----------
    max_depth : int
        Controls tree depth of the generated antiderivative ``F``.
    seed : int or None
        If not None, seed the RNG before generation for reproducibility.

    Returns
    -------
    tuple[sympy.Expr, sympy.Expr]
        ``(f, F)`` satisfying ``f == diff(F, x)``.
    """
    if seed is not None:
        random.seed(seed)
    for _ in range(_max_attempts):
        F = generate_expression(max_depth=max_depth)
        f = sympy.diff(F, x)

        # Reject trivial derivatives.
        if f == 0:
            continue
        if x not in f.free_symbols:
            continue

        # Verify f also roundtrips through prefix.
        if not _is_prefix_safe(f):
            continue

        return (f, F)

    # Fallback: x**2 with antiderivative x**3/3
    F_fallback = x ** 2
    f_fallback = sympy.diff(F_fallback, x)
    return (f_fallback, F_fallback)
