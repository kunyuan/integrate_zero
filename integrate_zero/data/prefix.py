"""Bidirectional conversion between SymPy expressions and prefix token lists.

This is the CORE serialization module for IntegrateZero.  Every component
that reads or writes mathematical expressions (generator, verifier, dataset,
model I/O, MCTS) depends on these two functions:

    sympy_to_prefix(expr) -> list[str]
    prefix_to_sympy(tokens: list[str]) -> sympy.Expr

The prefix (Polish) notation is an unambiguous, parenthesis-free
representation where each operator precedes its operands.

Key design decisions:
 - SymPy's n-ary Add/Mul are folded to binary left-associative trees.
 - SymPy encodes  -x      as  Mul(-1, x)         ->  ["neg", "x"]
 - SymPy encodes  x - y   as  Add(x, Mul(-1, y)) ->  ["sub", "x", "y"]
 - SymPy encodes  x / y   as  Mul(x, Pow(y, -1)) ->  ["div", "x", "y"]
 - SymPy encodes  sqrt(x) as  Pow(x, 1/2)        ->  ["sqrt", "x"]
 - Integral(f, x) is encoded as ["INT", ...f_tokens..., "x"]
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Type

import sympy
from sympy import (
    Add,
    E,
    Expr,
    Integer,
    Integral,
    Mul,
    Pow,
    Rational,
    Symbol,
    pi,
)
from sympy.core.numbers import (
    Exp1,
    Pi,
)

# ---------------------------------------------------------------------------
# SymPy function -> prefix token mapping
# ---------------------------------------------------------------------------

_SYMPY_FUNC_TO_TOKEN: Dict[Type, str] = {
    sympy.sin: "sin",
    sympy.cos: "cos",
    sympy.tan: "tan",
    sympy.exp: "exp",
    sympy.log: "log",
    sympy.asin: "arcsin",
    sympy.acos: "arccos",
    sympy.atan: "arctan",
    sympy.sinh: "sinh",
    sympy.cosh: "cosh",
    sympy.tanh: "tanh",
}

_TOKEN_TO_SYMPY_FUNC = {v: k for k, v in _SYMPY_FUNC_TO_TOKEN.items()}

# Symbols used in the vocabulary.
# Variables:
_VARIABLE_NAMES = {"x"}
# Real parameters:
_REAL_PARAM_NAMES = {"a", "b", "c", "d"}
# Integer parameters:
_INTEGER_PARAM_NAMES = {"k", "l", "m", "n"}

# Valid integer constant range
_MIN_INT = -25
_MAX_INT = 25


# ===================================================================
# sympy_to_prefix
# ===================================================================


def sympy_to_prefix(expr: Expr) -> List[str]:
    """Convert a SymPy expression to a prefix (Polish notation) token list.

    Parameters
    ----------
    expr : sympy.Expr
        A SymPy expression built from the supported vocabulary.

    Returns
    -------
    list[str]
        Tokens in prefix order.  Each token is a string present in the
        IntegrateZero vocabulary.

    Raises
    ------
    ValueError
        If the expression contains unsupported constructs.
    """
    return _to_prefix(expr)


def _to_prefix(expr: Expr) -> List[str]:
    """Recursive implementation of sympy_to_prefix."""

    # ------------------------------------------------------------------
    # 1. Named constants: pi, E (Euler's number)
    # ------------------------------------------------------------------
    if isinstance(expr, Pi):
        return ["pi"]
    if isinstance(expr, Exp1):
        return ["e"]

    # ------------------------------------------------------------------
    # 2. Numeric constants: integers (and negative integers)
    # ------------------------------------------------------------------
    if isinstance(expr, Integer):
        val = int(expr)
        if _MIN_INT <= val <= _MAX_INT:
            return [str(val)]
        raise ValueError(
            f"Integer {val} is outside the vocabulary range "
            f"[{_MIN_INT}, {_MAX_INT}]"
        )

    # ------------------------------------------------------------------
    # 3. Rational numbers (non-integer): encode as div(p, q)
    # ------------------------------------------------------------------
    if isinstance(expr, Rational):
        p, q = expr.p, expr.q
        return ["div", str(p), str(q)]

    # ------------------------------------------------------------------
    # 4. Symbols: x, a, b, c, d, k, l, m, n
    # ------------------------------------------------------------------
    if isinstance(expr, Symbol):
        name = expr.name
        if name in _VARIABLE_NAMES | _REAL_PARAM_NAMES | _INTEGER_PARAM_NAMES:
            return [name]
        raise ValueError(f"Symbol {name!r} is not in the vocabulary")

    # ------------------------------------------------------------------
    # 5. Integral: Integral(f, x) -> ["INT", ...f_tokens..., "x"]
    # ------------------------------------------------------------------
    if isinstance(expr, Integral):
        # We only support indefinite integrals over a single variable.
        integrand = expr.args[0]
        limits = expr.args[1]  # SymPy Tuple, e.g. (x,) for indefinite
        if len(limits) != 1:
            raise ValueError(
                f"Only indefinite integrals are supported, got limits={limits}"
            )
        var = limits[0]
        return ["INT"] + _to_prefix(integrand) + [var.name]

    # ------------------------------------------------------------------
    # 6. Addition: Add(...)
    #    Handle subtraction: Add(x, Mul(-1, y)) -> sub x y
    # ------------------------------------------------------------------
    if isinstance(expr, Add):
        return _add_to_prefix(expr)

    # ------------------------------------------------------------------
    # 7. Multiplication: Mul(...)
    #    Handle negation, division, negative coefficients
    # ------------------------------------------------------------------
    if isinstance(expr, Mul):
        return _mul_to_prefix(expr)

    # ------------------------------------------------------------------
    # 8. Power: Pow(base, exp)
    #    Handle sqrt: Pow(x, 1/2) -> sqrt x
    #    Handle division: Pow(x, -1) -> handled in Mul context
    # ------------------------------------------------------------------
    if isinstance(expr, Pow):
        return _pow_to_prefix(expr)

    # ------------------------------------------------------------------
    # 9. Known functions: sin, cos, tan, exp, log, ...
    # ------------------------------------------------------------------
    func_type = type(expr)
    if func_type in _SYMPY_FUNC_TO_TOKEN:
        token = _SYMPY_FUNC_TO_TOKEN[func_type]
        assert len(expr.args) == 1
        return [token] + _to_prefix(expr.args[0])

    raise ValueError(
        f"Unsupported expression type: {type(expr).__name__} "
        f"(expr = {expr})"
    )


# -------------------------------------------------------------------
# Add handler
# -------------------------------------------------------------------


def _add_to_prefix(expr: Add) -> List[str]:
    """Convert Add(...) to prefix tokens with subtraction detection.

    SymPy represents ``x - y`` as ``Add(x, Mul(-1, y))``.
    We detect terms of the form ``Mul(-1, stuff)`` and emit ``sub``
    instead of ``add`` + ``neg``.

    For n-ary sums we left-fold: Add(a, b, c) -> add(add(a, b), c).

    Positive terms are placed before negative terms so that we can use
    ``sub`` instead of ``neg`` + ``add`` whenever possible.
    """
    # Classify each term by sign using as_ordered_terms for determinism.
    pos_ordered: List[Expr] = []
    neg_ordered: List[Expr] = []

    for t in expr.as_ordered_terms():
        if _is_negative_term(t):
            neg_ordered.append(_negate(t))
        else:
            pos_ordered.append(t)

    # Build the final list: positive terms first, then negatives.
    # This ensures the first term is positive when possible, and we
    # can emit "sub" for all subsequent negative terms.
    ordered: List[Tuple[str, Expr]] = []
    for t in pos_ordered:
        ordered.append(("add", t))
    for t in neg_ordered:
        ordered.append(("sub", t))

    if not ordered:
        return ["0"]

    # Start with the first term (no operator prefix).
    first_op, first_term = ordered[0]
    if first_op == "sub":
        # All terms are negative.  Emit neg(first_term).
        tokens = ["neg"] + _to_prefix(first_term)
    else:
        tokens = _to_prefix(first_term)

    # Fold remaining terms.
    for op, term in ordered[1:]:
        tokens = [op] + tokens + _to_prefix(term)

    return tokens


def _is_negative_term(expr: Expr) -> bool:
    """Return True if *expr* should be treated as a subtracted term.

    This detects Mul(-1, ...) and negative integer coefficients.
    """
    if isinstance(expr, Mul):
        coeff, _ = expr.as_coeff_Mul()
        if coeff.is_negative:
            return True
    if isinstance(expr, Integer) and int(expr) < 0:
        return True
    if isinstance(expr, Rational) and expr.is_negative:
        return True
    return False


def _negate(expr: Expr) -> Expr:
    """Return the positive counterpart of a negative term.

    For Mul(-1, x) -> x, Mul(-3, x) -> Mul(3, x), Integer(-5) -> Integer(5).
    """
    return -expr


# -------------------------------------------------------------------
# Mul handler
# -------------------------------------------------------------------


def _mul_to_prefix(expr: Mul) -> List[str]:
    """Convert Mul(...) to prefix tokens with negation/division detection.

    SymPy encodings:
        -x       -> Mul(-1, x)             -> ["neg", "x"]
        x / y    -> Mul(x, Pow(y, -1))     -> ["div", "x", "y"]
        -x / y   -> Mul(-1, x, Pow(y, -1)) -> ["neg", "div", "x", "y"]
        -3 * x   -> Mul(-3, x)             -> ["mul", "-3", "x"]
                     BUT if -3 is outside vocab, use neg(mul(3, x)).

    For n-ary products we left-fold: Mul(a, b, c) -> mul(mul(a, b), c).
    """
    # Extract the numeric coefficient and the rest of the product.
    # e.g., -3*x -> coeff=-3, rest=x
    #       -x   -> coeff=-1, rest=x
    #       x/a  -> coeff=1, rest=x/a (but as_coeff_Mul may not split Pow)
    coeff, rest = expr.as_coeff_Mul()

    # Handle pure negation: Mul(-1, x) -> ["neg", "x"]
    if coeff == sympy.S.NegativeOne:
        return ["neg"] + _to_prefix(rest)

    # Handle negative coefficient that fits in vocabulary:
    # Mul(-3, x) -> ["mul", "-3", "x"]
    # Handle negative coefficient outside vocabulary:
    # Mul(-30, x) -> ["neg", "mul", "30", "x"]
    if isinstance(coeff, Integer) and int(coeff) < 0:
        neg_val = int(coeff)
        if _MIN_INT <= neg_val <= _MAX_INT:
            # Treat the negative coefficient as one factor.
            # Build factor list: [Integer(neg_val)] + non-coeff factors.
            factors = [Integer(neg_val)] + _non_coeff_factors(rest)
            return _factors_to_prefix(factors, expr)
        else:
            # Use neg(mul(abs(coeff), rest))
            return ["neg"] + _to_prefix(-expr)

    # Positive or unit coefficient.
    # Build the factor list from the original expression's factors,
    # but using as_coeff_Mul to keep the coefficient as one unit.
    factors: List[Expr] = []
    if isinstance(coeff, Integer) and int(coeff) != 1:
        factors.append(coeff)
    elif isinstance(coeff, Rational) and coeff != sympy.S.One:
        factors.append(coeff)
    factors.extend(_non_coeff_factors(rest))

    return _factors_to_prefix(factors, expr)


def _non_coeff_factors(expr: Expr) -> List[Expr]:
    """Get the non-coefficient factors of *expr* in ordered form.

    If *expr* is a Mul, returns its ordered factors.
    Otherwise returns [expr].
    """
    if isinstance(expr, Mul):
        return list(expr.as_ordered_factors())
    if expr == sympy.S.One:
        return []
    return [expr]


def _factors_to_prefix(factors: List[Expr], original: Expr) -> List[str]:
    """Convert a list of factors to prefix tokens, handling division.

    Separates numerator factors from denominator factors (Pow(x, -1)),
    then left-folds each group, wrapping with div if needed.
    """
    numer_factors: List[Expr] = []
    denom_factors: List[Expr] = []

    for arg in factors:
        if isinstance(arg, Pow) and _is_inverse_power(arg):
            denom_factors.append(arg.args[0])
        else:
            numer_factors.append(arg)

    # Build numerator token list (left-fold mul).
    if not numer_factors:
        numer_tokens = ["1"]
    elif len(numer_factors) == 1:
        numer_tokens = _to_prefix(numer_factors[0])
    else:
        numer_tokens = _to_prefix(numer_factors[0])
        for factor in numer_factors[1:]:
            numer_tokens = ["mul"] + numer_tokens + _to_prefix(factor)

    # Build denominator and wrap with div.
    if not denom_factors:
        return numer_tokens
    elif len(denom_factors) == 1:
        denom_tokens = _to_prefix(denom_factors[0])
    else:
        denom_tokens = _to_prefix(denom_factors[0])
        for factor in denom_factors[1:]:
            denom_tokens = ["mul"] + denom_tokens + _to_prefix(factor)

    return ["div"] + numer_tokens + denom_tokens


def _is_inverse_power(expr: Pow) -> bool:
    """Return True if expr is Pow(base, -1) i.e. a reciprocal."""
    base, exp = expr.args
    if isinstance(exp, Integer) and int(exp) == -1:
        return True
    if exp == sympy.S.NegativeOne:
        return True
    return False


# -------------------------------------------------------------------
# Pow handler
# -------------------------------------------------------------------


def _pow_to_prefix(expr: Pow) -> List[str]:
    """Convert Pow(base, exp) to prefix tokens.

    Special cases:
        Pow(x, 1/2)  -> ["sqrt", "x"]
        Pow(x, -1)   -> ["div", "1", "x"]  (standalone, not inside Mul)
        Pow(x, -1/2) -> ["div", "1", "sqrt", "x"]
    """
    base, exp = expr.args

    # sqrt: Pow(x, 1/2)
    if exp == sympy.Rational(1, 2):
        return ["sqrt"] + _to_prefix(base)

    # Reciprocal: Pow(x, -1).  When encountered standalone (not inside Mul).
    if exp == sympy.S.NegativeOne:
        return ["div", "1"] + _to_prefix(base)

    # Negative half: Pow(x, -1/2) -> div 1 sqrt x
    if exp == sympy.Rational(-1, 2):
        return ["div", "1"] + ["sqrt"] + _to_prefix(base)

    # General power: pow(base, exp)
    return ["pow"] + _to_prefix(base) + _to_prefix(exp)


# ===================================================================
# prefix_to_sympy
# ===================================================================

# Arity table for parsing.
_ARITY: Dict[str, int] = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "div": 2,
    "pow": 2,
    "neg": 1,
    "sin": 1,
    "cos": 1,
    "tan": 1,
    "exp": 1,
    "log": 1,
    "sqrt": 1,
    "arcsin": 1,
    "arccos": 1,
    "arctan": 1,
    "sinh": 1,
    "cosh": 1,
    "tanh": 1,
    "INT": 2,
}


def prefix_to_sympy(tokens: List[str]) -> Expr:
    """Convert a prefix (Polish notation) token list to a SymPy expression.

    Parameters
    ----------
    tokens : list[str]
        Prefix token list (as produced by :func:`sympy_to_prefix`).

    Returns
    -------
    sympy.Expr
        The reconstructed SymPy expression.

    Raises
    ------
    ValueError
        If the token list is malformed or contains unknown tokens.
    """
    pos, result = _parse(tokens, 0)
    if pos != len(tokens):
        raise ValueError(
            f"Extra tokens after position {pos}: {tokens[pos:]}"
        )
    return result


def _parse(tokens: List[str], pos: int) -> Tuple[int, Expr]:
    """Parse one expression starting at *pos*, return (new_pos, expr)."""
    if pos >= len(tokens):
        raise ValueError(f"Unexpected end of tokens at position {pos}")

    token = tokens[pos]
    pos += 1

    # ------------------------------------------------------------------
    # Operators and functions with known arity
    # ------------------------------------------------------------------
    if token in _ARITY:
        arity = _ARITY[token]
        children: List[Expr] = []
        for _ in range(arity):
            pos, child = _parse(tokens, pos)
            children.append(child)
        return pos, _build_expr(token, children)

    # ------------------------------------------------------------------
    # Leaf tokens: constants, variables, parameters
    # ------------------------------------------------------------------
    return pos, _token_to_leaf(token)


def _build_expr(token: str, children: List[Expr]) -> Expr:
    """Build a SymPy expression from an operator/function token and children."""
    if token == "add":
        return children[0] + children[1]
    if token == "sub":
        return children[0] - children[1]
    if token == "mul":
        return children[0] * children[1]
    if token == "div":
        return children[0] / children[1]
    if token == "pow":
        return children[0] ** children[1]
    if token == "neg":
        return -children[0]
    if token == "sqrt":
        return sympy.sqrt(children[0])
    if token == "INT":
        return Integral(children[0], children[1])

    # Named functions
    if token in _TOKEN_TO_SYMPY_FUNC:
        func = _TOKEN_TO_SYMPY_FUNC[token]
        return func(children[0])

    raise ValueError(f"Unknown operator/function token: {token!r}")


def _token_to_leaf(token: str) -> Expr:
    """Convert a leaf token to a SymPy atom (Symbol, Integer, or constant)."""
    # Named constants
    if token == "pi":
        return pi
    if token == "e":
        return E

    # Integer constants (including negative like "-5")
    try:
        val = int(token)
        return Integer(val)
    except ValueError:
        pass

    # Symbols
    if token in _VARIABLE_NAMES:
        return Symbol(token)
    if token in _REAL_PARAM_NAMES:
        return Symbol(token, real=True)
    if token in _INTEGER_PARAM_NAMES:
        return Symbol(token, integer=True)

    raise ValueError(f"Unknown leaf token: {token!r}")
