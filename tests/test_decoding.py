"""Tests for arity-constrained decoding (ArityMask)."""

from integrate_zero.model.decoding import ArityMask
from integrate_zero.data.vocabulary import Vocabulary


def test_arity_mask_forces_eos_when_stack_empty():
    vocab = Vocabulary()
    mask = ArityMask(vocab)
    tokens = ["x"]  # complete expression
    allowed = mask.get_allowed_tokens(tokens)
    eos_id = vocab.token_to_id("EOS")
    assert eos_id in allowed


def test_arity_mask_blocks_eos_when_stack_not_empty():
    vocab = Vocabulary()
    mask = ArityMask(vocab)
    tokens = ["add", "x"]  # need one more child
    allowed = mask.get_allowed_tokens(tokens)
    eos_id = vocab.token_to_id("EOS")
    assert eos_id not in allowed


def test_arity_mask_after_binary_op():
    vocab = Vocabulary()
    mask = ArityMask(vocab)
    tokens = ["add"]  # need 2 children
    allowed = mask.get_allowed_tokens(tokens)
    eos_id = vocab.token_to_id("EOS")
    assert eos_id not in allowed
    assert vocab.token_to_id("x") in allowed
    assert vocab.token_to_id("sin") in allowed


def test_complete_expression_valid():
    vocab = Vocabulary()
    mask = ArityMask(vocab)
    tokens = ["add", "sin", "x", "x"]  # add(sin(x), x) - complete
    allowed = mask.get_allowed_tokens(tokens)
    eos_id = vocab.token_to_id("EOS")
    assert eos_id in allowed


def test_special_tokens_never_allowed():
    """PAD, BOS, SEP should never appear in allowed tokens during generation."""
    vocab = Vocabulary()
    mask = ArityMask(vocab)
    pad_id = vocab.token_to_id("PAD")
    bos_id = vocab.token_to_id("BOS")
    sep_id = vocab.token_to_id("SEP")

    # When expression is incomplete
    allowed_incomplete = mask.get_allowed_tokens(["add"])
    assert pad_id not in allowed_incomplete
    assert bos_id not in allowed_incomplete
    assert sep_id not in allowed_incomplete

    # When expression is complete
    allowed_complete = mask.get_allowed_tokens(["x"])
    assert pad_id not in allowed_complete
    assert bos_id not in allowed_complete
    assert sep_id not in allowed_complete


def test_empty_tokens_allows_all_content():
    """With no tokens generated yet (remaining=1), all content tokens should
    be allowed and EOS should be blocked."""
    vocab = Vocabulary()
    mask = ArityMask(vocab)
    allowed = mask.get_allowed_tokens([])
    eos_id = vocab.token_to_id("EOS")
    assert eos_id not in allowed
    # Should allow operators, functions, variables, constants
    assert vocab.token_to_id("add") in allowed
    assert vocab.token_to_id("sin") in allowed
    assert vocab.token_to_id("x") in allowed
    assert vocab.token_to_id("1") in allowed


def test_nested_unary_ops():
    """sin(sin(x)) should be a complete expression."""
    vocab = Vocabulary()
    mask = ArityMask(vocab)
    # sin -> remaining = 1; sin -> remaining = 1; x -> remaining = 0
    tokens = ["sin", "sin", "x"]
    allowed = mask.get_allowed_tokens(tokens)
    eos_id = vocab.token_to_id("EOS")
    assert eos_id in allowed


def test_deep_binary_expression():
    """add(add(x, x), x) should be complete."""
    vocab = Vocabulary()
    mask = ArityMask(vocab)
    # add -> rem=2; add -> rem=3; x -> rem=2; x -> rem=1; x -> rem=0
    tokens = ["add", "add", "x", "x", "x"]
    allowed = mask.get_allowed_tokens(tokens)
    eos_id = vocab.token_to_id("EOS")
    assert eos_id in allowed


def test_only_eos_allowed_when_complete():
    """When the expression is complete (remaining==0), only EOS should be allowed."""
    vocab = Vocabulary()
    mask = ArityMask(vocab)
    tokens = ["x"]
    allowed = mask.get_allowed_tokens(tokens)
    eos_id = vocab.token_to_id("EOS")
    # EOS must be the only allowed token
    assert allowed == {eos_id}


def test_numeric_constant_is_leaf():
    """Numeric constants like '3' have arity 0, so they complete a slot."""
    vocab = Vocabulary()
    mask = ArityMask(vocab)
    tokens = ["add", "3", "5"]  # add(3, 5) — complete
    allowed = mask.get_allowed_tokens(tokens)
    eos_id = vocab.token_to_id("EOS")
    assert eos_id in allowed
