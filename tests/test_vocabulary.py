"""Tests for the Vocabulary class."""

import pytest

from integrate_zero.data.vocabulary import Vocabulary


def test_special_tokens_have_fixed_ids():
    vocab = Vocabulary()
    assert vocab.token_to_id("PAD") == 0
    assert vocab.token_to_id("BOS") == 1
    assert vocab.token_to_id("EOS") == 2
    assert vocab.token_to_id("SEP") == 3


def test_operators_in_vocab():
    vocab = Vocabulary()
    for op in ["add", "sub", "mul", "div", "pow", "neg"]:
        assert vocab.token_to_id(op) is not None


def test_functions_in_vocab():
    vocab = Vocabulary()
    for fn in [
        "sin", "cos", "tan", "exp", "log", "sqrt",
        "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh",
    ]:
        assert vocab.token_to_id(fn) is not None


def test_int_token():
    vocab = Vocabulary()
    assert vocab.token_to_id("INT") is not None


def test_variables_and_params():
    vocab = Vocabulary()
    assert vocab.token_to_id("x") is not None
    for p in ["a", "b", "c", "d"]:
        assert vocab.token_to_id(p) is not None
    for p in ["k", "l", "m", "n"]:
        assert vocab.token_to_id(p) is not None


def test_numeric_constants():
    vocab = Vocabulary()
    for i in range(-10, 11):
        assert vocab.token_to_id(str(i)) is not None
    assert vocab.token_to_id("pi") is not None
    assert vocab.token_to_id("e") is not None


def test_roundtrip():
    vocab = Vocabulary()
    for tok in ["add", "sin", "x", "INT", "3", "pi"]:
        assert vocab.id_to_token(vocab.token_to_id(tok)) == tok


def test_arity():
    vocab = Vocabulary()
    assert vocab.arity("add") == 2
    assert vocab.arity("sin") == 1
    assert vocab.arity("INT") == 2
    assert vocab.arity("x") == 0
    assert vocab.arity("3") == 0
    assert vocab.arity("pi") == 0


def test_vocab_size():
    vocab = Vocabulary()
    assert 80 <= len(vocab) <= 100


def test_l_and_1_are_distinct():
    """Lowercase letter 'l' and digit '1' must be separate tokens."""
    vocab = Vocabulary()
    id_l = vocab.token_to_id("l")
    id_1 = vocab.token_to_id("1")
    assert id_l is not None
    assert id_1 is not None
    assert id_l != id_1


def test_neg_is_unary():
    vocab = Vocabulary()
    assert vocab.arity("neg") == 1


def test_binary_operators_arity():
    vocab = Vocabulary()
    for op in ["add", "sub", "mul", "div", "pow"]:
        assert vocab.arity(op) == 2


def test_all_functions_are_unary():
    vocab = Vocabulary()
    for fn in [
        "sin", "cos", "tan", "exp", "log", "sqrt",
        "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh",
    ]:
        assert vocab.arity(fn) == 1


def test_special_tokens_arity():
    vocab = Vocabulary()
    for tok in ["PAD", "BOS", "EOS", "SEP"]:
        assert vocab.arity(tok) == 0


def test_id_to_token_invalid_returns_none():
    vocab = Vocabulary()
    assert vocab.id_to_token(-1) is None
    assert vocab.id_to_token(9999) is None


def test_token_to_id_unknown_returns_none():
    vocab = Vocabulary()
    assert vocab.token_to_id("UNKNOWN_TOKEN") is None


def test_negative_number_strings():
    """Negative numbers like '-10', '-1' are valid tokens with arity 0."""
    vocab = Vocabulary()
    for i in range(-10, 0):
        tok = str(i)
        assert vocab.token_to_id(tok) is not None, f"Missing token: {tok}"
        assert vocab.arity(tok) == 0, f"Wrong arity for {tok}"


def test_all_ids_are_unique():
    """Every token must have a unique ID."""
    vocab = Vocabulary()
    seen_ids = set()
    for i in range(len(vocab)):
        tok = vocab.id_to_token(i)
        assert tok is not None, f"No token for id {i}"
        assert i not in seen_ids, f"Duplicate id {i}"
        seen_ids.add(i)


def test_ids_are_contiguous():
    """IDs should be 0, 1, 2, ..., len(vocab)-1."""
    vocab = Vocabulary()
    for i in range(len(vocab)):
        assert vocab.id_to_token(i) is not None, f"Gap at id {i}"
