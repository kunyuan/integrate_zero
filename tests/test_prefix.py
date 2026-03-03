"""Tests for prefix expression <-> SymPy bidirectional conversion."""

import sympy
import pytest

from integrate_zero.data.prefix import sympy_to_prefix, prefix_to_sympy

x = sympy.Symbol("x")
a = sympy.Symbol("a", real=True)
n = sympy.Symbol("n", integer=True)


def test_simple_variable():
    assert sympy_to_prefix(x) == ["x"]
    assert prefix_to_sympy(["x"]) == x


def test_integer_constant():
    assert sympy_to_prefix(sympy.Integer(3)) == ["3"]
    assert prefix_to_sympy(["3"]) == sympy.Integer(3)


def test_addition():
    expr = x + sympy.Integer(1)
    tokens = sympy_to_prefix(expr)
    assert tokens == ["add", "x", "1"]
    assert prefix_to_sympy(tokens) == expr


def test_sin():
    expr = sympy.sin(x)
    tokens = sympy_to_prefix(expr)
    assert tokens == ["sin", "x"]
    assert prefix_to_sympy(tokens) == expr


def test_nested():
    expr = sympy.sin(x**2 + 1)
    tokens = sympy_to_prefix(expr)
    reconstructed = prefix_to_sympy(tokens)
    assert sympy.simplify(reconstructed - expr) == 0


def test_with_integral():
    expr = sympy.Integral(sympy.sin(x), x)
    tokens = sympy_to_prefix(expr)
    assert tokens[0] == "INT"
    assert tokens[-1] == "x"
    assert prefix_to_sympy(tokens) == expr


def test_with_parameter():
    expr = a * x**n
    tokens = sympy_to_prefix(expr)
    reconstructed = prefix_to_sympy(tokens)
    assert sympy.simplify(reconstructed - expr) == 0


def test_roundtrip_complex():
    expr = x * sympy.sin(x) + sympy.cos(x)
    tokens = sympy_to_prefix(expr)
    reconstructed = prefix_to_sympy(tokens)
    assert sympy.simplify(reconstructed - expr) == 0


def test_negation():
    expr = -x
    tokens = sympy_to_prefix(expr)
    reconstructed = prefix_to_sympy(tokens)
    assert sympy.simplify(reconstructed - expr) == 0


def test_pi_and_e():
    expr = sympy.pi * x + sympy.E
    tokens = sympy_to_prefix(expr)
    reconstructed = prefix_to_sympy(tokens)
    assert sympy.simplify(reconstructed - expr) == 0


# ---------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------


def test_subtraction():
    """x - y should emit sub tokens, not add with neg."""
    expr = x - a
    tokens = sympy_to_prefix(expr)
    reconstructed = prefix_to_sympy(tokens)
    assert sympy.simplify(reconstructed - expr) == 0


def test_division():
    """x / y should emit div tokens, not mul with pow -1."""
    expr = x / a
    tokens = sympy_to_prefix(expr)
    reconstructed = prefix_to_sympy(tokens)
    assert sympy.simplify(reconstructed - expr) == 0


def test_zero():
    assert sympy_to_prefix(sympy.Integer(0)) == ["0"]
    assert prefix_to_sympy(["0"]) == sympy.Integer(0)


def test_one():
    assert sympy_to_prefix(sympy.Integer(1)) == ["1"]
    assert prefix_to_sympy(["1"]) == sympy.Integer(1)


def test_negative_integer():
    assert sympy_to_prefix(sympy.Integer(-5)) == ["-5"]
    assert prefix_to_sympy(["-5"]) == sympy.Integer(-5)


def test_nary_addition_folds_left():
    """Add(a, b, c) -> add add a b c (left-fold)."""
    expr = x + a + n
    tokens = sympy_to_prefix(expr)
    reconstructed = prefix_to_sympy(tokens)
    assert sympy.simplify(reconstructed - expr) == 0
    # Should have exactly 2 "add" tokens for 3 operands
    assert tokens.count("add") == 2


def test_exp():
    expr = sympy.exp(x)
    tokens = sympy_to_prefix(expr)
    assert tokens == ["exp", "x"]
    assert prefix_to_sympy(tokens) == expr


def test_log():
    expr = sympy.log(x)
    tokens = sympy_to_prefix(expr)
    assert tokens == ["log", "x"]
    assert prefix_to_sympy(tokens) == expr


def test_sqrt():
    expr = sympy.sqrt(x)
    tokens = sympy_to_prefix(expr)
    assert tokens == ["sqrt", "x"]
    assert prefix_to_sympy(tokens) == expr


def test_power():
    expr = x**sympy.Integer(2)
    tokens = sympy_to_prefix(expr)
    assert tokens == ["pow", "x", "2"]
    assert prefix_to_sympy(tokens) == expr


def test_cos():
    expr = sympy.cos(x)
    tokens = sympy_to_prefix(expr)
    assert tokens == ["cos", "x"]
    assert prefix_to_sympy(tokens) == expr


def test_tan():
    expr = sympy.tan(x)
    tokens = sympy_to_prefix(expr)
    assert tokens == ["tan", "x"]
    assert prefix_to_sympy(tokens) == expr


def test_arcsin():
    expr = sympy.asin(x)
    tokens = sympy_to_prefix(expr)
    assert tokens == ["arcsin", "x"]
    assert prefix_to_sympy(tokens) == expr


def test_arccos():
    expr = sympy.acos(x)
    tokens = sympy_to_prefix(expr)
    assert tokens == ["arccos", "x"]
    assert prefix_to_sympy(tokens) == expr


def test_arctan():
    expr = sympy.atan(x)
    tokens = sympy_to_prefix(expr)
    assert tokens == ["arctan", "x"]
    assert prefix_to_sympy(tokens) == expr


def test_sinh():
    expr = sympy.sinh(x)
    tokens = sympy_to_prefix(expr)
    assert tokens == ["sinh", "x"]
    assert prefix_to_sympy(tokens) == expr


def test_cosh():
    expr = sympy.cosh(x)
    tokens = sympy_to_prefix(expr)
    assert tokens == ["cosh", "x"]
    assert prefix_to_sympy(tokens) == expr


def test_tanh():
    expr = sympy.tanh(x)
    tokens = sympy_to_prefix(expr)
    assert tokens == ["tanh", "x"]
    assert prefix_to_sympy(tokens) == expr


def test_neg_emits_neg_token():
    """Negation of a variable should use the neg token."""
    expr = -x
    tokens = sympy_to_prefix(expr)
    assert tokens == ["neg", "x"]


def test_mul_neg_one_is_neg():
    """Mul(-1, x) is SymPy's internal representation of -x; should be neg."""
    expr = sympy.Mul(sympy.Integer(-1), x)
    tokens = sympy_to_prefix(expr)
    assert tokens == ["neg", "x"]


def test_division_tokens():
    """x / a should produce div tokens."""
    expr = x / a
    tokens = sympy_to_prefix(expr)
    assert "div" in tokens


def test_subtraction_tokens():
    """x - a should produce sub tokens."""
    expr = x - a
    tokens = sympy_to_prefix(expr)
    assert "sub" in tokens


def test_pi_constant():
    assert sympy_to_prefix(sympy.pi) == ["pi"]
    assert prefix_to_sympy(["pi"]) == sympy.pi


def test_e_constant():
    assert sympy_to_prefix(sympy.E) == ["e"]
    assert prefix_to_sympy(["e"]) == sympy.E


def test_roundtrip_deeply_nested():
    """Test a deeply nested expression roundtrips correctly."""
    expr = sympy.sin(sympy.cos(sympy.exp(x**2 + sympy.Integer(1))))
    tokens = sympy_to_prefix(expr)
    reconstructed = prefix_to_sympy(tokens)
    assert sympy.simplify(reconstructed - expr) == 0


def test_integer_param_n():
    """Integer parameter n should roundtrip."""
    expr = x**n
    tokens = sympy_to_prefix(expr)
    reconstructed = prefix_to_sympy(tokens)
    assert sympy.simplify(reconstructed - expr) == 0


def test_real_param_roundtrip():
    """Real parameters a, b, c, d should all roundtrip."""
    b = sympy.Symbol("b", real=True)
    c = sympy.Symbol("c", real=True)
    expr = a * x**2 + b * x + c
    tokens = sympy_to_prefix(expr)
    reconstructed = prefix_to_sympy(tokens)
    assert sympy.simplify(reconstructed - expr) == 0


def test_negative_coefficient():
    """Expression like -3*x should roundtrip."""
    expr = sympy.Integer(-3) * x
    tokens = sympy_to_prefix(expr)
    reconstructed = prefix_to_sympy(tokens)
    assert sympy.simplify(reconstructed - expr) == 0


def test_integral_complex():
    """Integral of a more complex expression."""
    expr = sympy.Integral(x**2 * sympy.sin(x), x)
    tokens = sympy_to_prefix(expr)
    assert tokens[0] == "INT"
    assert tokens[-1] == "x"
    reconstructed = prefix_to_sympy(tokens)
    assert reconstructed == expr
