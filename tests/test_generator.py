import sympy
from integrate_zero.data.generator import generate_expression, generate_training_pair
from integrate_zero.data.prefix import sympy_to_prefix, prefix_to_sympy

x = sympy.Symbol("x")

def test_generate_expression_returns_sympy():
    expr = generate_expression(max_depth=3)
    assert isinstance(expr, sympy.Basic)
    assert x in expr.free_symbols

def test_generate_expression_depth():
    expr = generate_expression(max_depth=2)
    tokens = sympy_to_prefix(expr)
    assert len(tokens) <= 15

def test_generate_training_pair():
    f, F = generate_training_pair(max_depth=3)
    dF = sympy.diff(F, x)
    assert sympy.simplify(dF - f) == 0

def test_generate_multiple_unique():
    pairs = [generate_training_pair(max_depth=3) for _ in range(20)]
    f_strs = [str(p[0]) for p in pairs]
    assert len(set(f_strs)) > 1

def test_generate_expression_roundtrips_through_prefix():
    for _ in range(10):
        expr = generate_expression(max_depth=3)
        tokens = sympy_to_prefix(expr)
        reconstructed = prefix_to_sympy(tokens)
        assert sympy.simplify(reconstructed - expr) == 0
