"""Verification module for integration step correctness.

This module provides the "referee" logic that determines whether a
model-generated expression is a valid transformation of the previous
expression. It supports three verdict types:

- IDENTITY: A mathematically equivalent rewrite (e.g., trig identity).
- INTEGRATION: A valid integration step (partial or complete).
- INVALID: The transformation is not mathematically justified.

Verification uses symbolic simplification first, falling back to
numerical sampling when symbolic methods are inconclusive.
"""

from __future__ import annotations

import enum
import random
import signal
from typing import Optional

import sympy
from sympy import Symbol, Integral, simplify, diff


class _Timeout:
    """Context manager that raises TimeoutError after *seconds*.

    Uses SIGALRM on Unix.  On platforms without SIGALRM the timeout
    is silently skipped (no protection, but no crash either).
    """

    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self._has_alarm = hasattr(signal, "SIGALRM")

    def _handler(self, signum, frame):
        raise TimeoutError("verification timed out")

    def __enter__(self):
        if self._has_alarm:
            self._old = signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._has_alarm:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old)
        return False


class StepType(enum.Enum):
    """Classification of a transformation step."""

    INTEGRATION = "integration"
    IDENTITY = "identity"
    INVALID = "invalid"


def is_terminal(expr: sympy.Expr) -> bool:
    """Check whether an expression contains no Integral symbols.

    An expression is terminal when it represents a closed-form result
    (no remaining unevaluated integrals).

    Parameters
    ----------
    expr : sympy.Expr
        The expression to check.

    Returns
    -------
    bool
        True if the expression contains no ``Integral`` sub-expressions.
    """
    return not expr.has(Integral)


def _symbolic_equal(a: sympy.Expr, b: sympy.Expr, x: Symbol) -> Optional[bool]:
    """Try to determine symbolic equality of two expressions.

    Returns True if symbolically equal, False if symbolically unequal,
    or None if the result is inconclusive.
    """
    diff_expr = simplify(a - b)
    if diff_expr == 0:
        return True
    # Also try expand + trigsimp for stubborn cases
    diff_expr2 = sympy.trigsimp(sympy.expand(a - b))
    if diff_expr2 == 0:
        return True
    return None


def _freeze_integrals(expr: sympy.Expr):
    """Replace all Integral sub-expressions with unique dummy symbols.

    This prevents SymPy from evaluating integrals during identity
    comparison. Returns (new_expr, substitution_dict).
    """
    integrals = list(expr.atoms(Integral))
    subs_map = {}
    for i, integ in enumerate(integrals):
        dummy = sympy.Dummy(f"_frozen_int_{i}")
        subs_map[integ] = dummy
    frozen = expr.subs(subs_map)
    return frozen, subs_map


def _identity_equal(a: sympy.Expr, b: sympy.Expr, x: Symbol) -> bool:
    """Check if A and B are identical up to algebraic/trig rewrites.

    This freezes all Integral sub-expressions to prevent SymPy from
    evaluating them, then checks symbolic equality of the frozen forms.
    This ensures that only genuine algebraic rewrites (not integration
    steps) are classified as IDENTITY.
    """
    # Collect integrals from both expressions
    a_integrals = list(a.atoms(Integral))
    b_integrals = list(b.atoms(Integral))

    # If the integral structures differ (e.g., A has Integral but B doesn't),
    # this cannot be an identity rewrite.
    if len(a_integrals) != len(b_integrals):
        return False

    # If neither expression has integrals, just compare directly
    if not a_integrals and not b_integrals:
        return _exprs_equal(a, b, x)

    # Both have integrals — try to match them.
    # For expressions like Integral(sin(x)*cos(x), x) vs Integral(sin(2x)/2, x),
    # the integrands differ but are equivalent. We need to:
    # 1. Check that the integrands are equivalent (identity rewrite of integrand)
    # 2. Check that the non-integral parts are equivalent

    # Simple case: both A and B are pure Integrals
    if isinstance(a, Integral) and isinstance(b, Integral):
        a_integrand, a_var = _extract_integrand_and_var(a)
        b_integrand, b_var = _extract_integrand_and_var(b)
        return _exprs_equal(a_integrand, b_integrand, x)

    # General case: try doit() on both and compare, but only if the
    # integral structures look similar. This handles compound expressions
    # where integrals appear as sub-expressions.
    # First try freezing integrals and comparing the algebraic structure.
    # We freeze matching pairs of integrals from A and B with the same dummy.
    # Try all permutations of matching (for small counts).
    if len(a_integrals) == len(b_integrals):
        from itertools import permutations

        for perm in permutations(range(len(b_integrals))):
            # Check if this pairing of integrals makes sense
            # (i.e., each paired integrand is equivalent)
            all_match = True
            subs_a = {}
            subs_b = {}
            for ai, bi in enumerate(perm):
                a_int = a_integrals[ai]
                b_int = b_integrals[bi]
                a_ig, _ = _extract_integrand_and_var(a_int)
                b_ig, _ = _extract_integrand_and_var(b_int)
                if not _exprs_equal(a_ig, b_ig, x):
                    all_match = False
                    break
                dummy = sympy.Dummy(f"_paired_int_{ai}")
                subs_a[a_int] = dummy
                subs_b[b_int] = dummy
            if all_match:
                frozen_a = a.subs(subs_a)
                frozen_b = b.subs(subs_b)
                if _exprs_equal(frozen_a, frozen_b, x):
                    return True

    return False


def _numerical_equal(
    a: sympy.Expr,
    b: sympy.Expr,
    x: Symbol,
    n_samples: int = 20,
    tol: float = 1e-8,
) -> bool:
    """Compare two expressions numerically by sampling random points.

    Generates ``n_samples`` random real values for *x* and evaluates
    both expressions, declaring equality if all sampled differences
    are within *tol*.

    Parameters
    ----------
    a, b : sympy.Expr
        Expressions to compare (must be in terms of *x*).
    x : Symbol
        The free variable.
    n_samples : int
        Number of random sample points.
    tol : float
        Absolute tolerance for declaring numerical equality.

    Returns
    -------
    bool
        True if the expressions agree at all sample points.
    """
    rng = random.Random(42)  # deterministic for reproducibility
    matches = 0
    evaluated = 0

    for _ in range(n_samples):
        # Sample from a moderate range to avoid overflow issues
        val = rng.uniform(-2.0, 2.0)
        try:
            a_val = complex(a.subs(x, val))
            b_val = complex(b.subs(x, val))
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            continue

        # Skip points where either expression is non-finite
        if any(
            v != v or abs(v) == float("inf")  # NaN or Inf check
            for v in (a_val.real, a_val.imag, b_val.real, b_val.imag)
        ):
            continue

        evaluated += 1
        if abs(a_val - b_val) < tol:
            matches += 1

    # Need at least a few valid sample points to declare equality
    if evaluated < 5:
        return False
    return matches == evaluated


def _exprs_equal(a: sympy.Expr, b: sympy.Expr, x: Symbol) -> bool:
    """Check equality of two expressions using symbolic + numerical fallback."""
    sym_result = _symbolic_equal(a, b, x)
    if sym_result is True:
        return True
    if sym_result is False:
        return False
    # Inconclusive symbolically — try numerical fallback
    return _numerical_equal(a, b, x)


def _extract_integrand_and_var(integral_expr: Integral):
    """Extract the integrand and variable from an Integral expression.

    Parameters
    ----------
    integral_expr : Integral
        A SymPy ``Integral`` object.

    Returns
    -------
    tuple[sympy.Expr, Symbol]
        The integrand and the integration variable.
    """
    integrand = integral_expr.args[0]
    # args[1] is (var,) or (var, lo, hi) for definite integrals
    var_info = integral_expr.args[1]
    if isinstance(var_info, tuple):
        var = var_info[0]
    else:
        var = var_info
    return integrand, var


def verify_step(
    A: sympy.Expr,
    B: sympy.Expr,
    x: Symbol = Symbol("x"),
    timeout: int = 5,
) -> StepType:
    """Verify a transformation step from expression A to expression B.

    The verification proceeds in order:

    1. **Identity check**: If ``simplify(A - B) == 0`` (or numerical
       fallback agrees), the step is an identity rewrite.
    2. **Integration check**:
       - If B is terminal (no ``Integral``): verify that
         ``d/dx(B) == integrand_of(A)``.
       - If B is non-terminal (has ``Integral``): evaluate both A and B
         via ``doit()`` and check equivalence.
    3. If neither check passes, the step is **invalid**.

    Parameters
    ----------
    A : sympy.Expr
        The source expression (typically contains an ``Integral``).
    B : sympy.Expr
        The target expression (the proposed transformation).
    x : Symbol
        The integration / differentiation variable.
    timeout : int
        Maximum seconds before declaring the step invalid (default: 5).

    Returns
    -------
    StepType
        The classification of the step.
    """
    try:
        with _Timeout(timeout):
            return _verify_step_inner(A, B, x)
    except TimeoutError:
        return StepType.INVALID


def _verify_step_inner(
    A: sympy.Expr,
    B: sympy.Expr,
    x: Symbol,
) -> StepType:
    """Core verification logic (no timeout guard)."""
    # --- 1. Identity rewrite check ---
    if _identity_equal(A, B, x):
        return StepType.IDENTITY

    # --- 2. Integration step check ---
    if is_terminal(B):
        # B has no integrals — check via differentiation
        # A must contain at least one Integral for this to be an integration step
        if A.has(Integral):
            # Extract the integrand from the outermost Integral in A
            # If A is itself an Integral, use it directly
            if isinstance(A, Integral):
                integrand, var = _extract_integrand_and_var(A)
            else:
                # A is a compound expression; find the Integral subexpression
                # For now, try evaluating A.doit() and comparing to B
                a_evaluated = A.doit()
                if _exprs_equal(a_evaluated, B, x):
                    return StepType.INTEGRATION
                return StepType.INVALID

            dB = diff(B, x)
            if _exprs_equal(dB, integrand, x):
                return StepType.INTEGRATION
    else:
        # B is non-terminal (still has Integral) — evaluate both via doit()
        if A.has(Integral):
            try:
                a_evaluated = A.doit()
                b_evaluated = B.doit()
                if _exprs_equal(a_evaluated, b_evaluated, x):
                    return StepType.INTEGRATION
            except Exception:
                pass

    # --- 3. Neither identity nor valid integration ---
    return StepType.INVALID
