"""Tests for integrate_zero.verify.verifier module."""

import sympy
from integrate_zero.verify.verifier import verify_step, is_terminal, StepType

x = sympy.Symbol("x")


def test_valid_integration_step():
    A = sympy.Integral(x * sympy.cos(x), x)
    B = x * sympy.sin(x) + sympy.cos(x)
    result = verify_step(A, B)
    assert result == StepType.INTEGRATION


def test_valid_partial_integration():
    A = sympy.Integral(x * sympy.cos(x), x)
    B = x * sympy.sin(x) - sympy.Integral(sympy.sin(x), x)
    result = verify_step(A, B)
    assert result == StepType.INTEGRATION


def test_identity_rewrite():
    A = sympy.Integral(sympy.sin(x) * sympy.cos(x), x)
    B = sympy.Integral(sympy.sin(2 * x) / 2, x)
    result = verify_step(A, B)
    assert result == StepType.IDENTITY


def test_invalid_step():
    A = sympy.Integral(sympy.sin(x), x)
    B = x**2
    result = verify_step(A, B)
    assert result == StepType.INVALID


def test_is_terminal():
    assert is_terminal(x * sympy.sin(x) + sympy.cos(x)) is True
    assert is_terminal(x * sympy.sin(x) - sympy.Integral(sympy.sin(x), x)) is False


def test_numerical_fallback():
    A = sympy.Integral(sympy.exp(x) * sympy.cos(x), x)
    B = sympy.exp(x) * (sympy.sin(x) + sympy.cos(x)) / 2
    result = verify_step(A, B)
    assert result == StepType.INTEGRATION
