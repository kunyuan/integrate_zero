"""Vocabulary and tokenizer for prefix-notation mathematical expressions.

This module defines the token vocabulary used throughout the IntegrateZero
system: prefix expression encoding/decoding, model input/output, and
arity-constrained decoding during MCTS.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Token definitions
#
# Each entry is (token_string, arity).
# Order matters: the position in the final list becomes the token ID.
# Special tokens MUST occupy IDs 0-3.
# ---------------------------------------------------------------------------

_SPECIAL: List[Tuple[str, int]] = [
    ("PAD", 0),  # id 0 — padding
    ("BOS", 0),  # id 1 — beginning of sequence
    ("EOS", 0),  # id 2 — end of sequence
    ("SEP", 0),  # id 3 — separator (e.g. between integrand and result)
]

_BINARY_OPS: List[Tuple[str, int]] = [
    ("add", 2),
    ("sub", 2),
    ("mul", 2),
    ("div", 2),
    ("pow", 2),
]

_UNARY_OPS: List[Tuple[str, int]] = [
    ("neg", 1),
]

_FUNCTIONS: List[Tuple[str, int]] = [
    ("sin", 1),
    ("cos", 1),
    ("tan", 1),
    ("exp", 1),
    ("log", 1),
    ("sqrt", 1),
    ("arcsin", 1),
    ("arccos", 1),
    ("arctan", 1),
    ("sinh", 1),
    ("cosh", 1),
    ("tanh", 1),
]

_INTEGRAL: List[Tuple[str, int]] = [
    ("INT", 2),  # prefix INT <integrand> <variable>
]

_VARIABLES: List[Tuple[str, int]] = [
    ("x", 0),  # integration variable
]

_REAL_PARAMS: List[Tuple[str, int]] = [
    ("a", 0),
    ("b", 0),
    ("c", 0),
    ("d", 0),
]

_INTEGER_PARAMS: List[Tuple[str, int]] = [
    ("k", 0),
    ("l", 0),  # lowercase L — distinct from digit "1"
    ("m", 0),
    ("n", 0),
]

# Integer constants from -25 to 25 (as strings).  The core range -10..10 is
# always required; extending to -25..25 gives a vocabulary of ~85 tokens,
# covering coefficients that commonly appear in generated expressions.
_NUMERIC_INTEGERS: List[Tuple[str, int]] = [
    (str(i), 0) for i in range(-25, 26)
]

_NAMED_CONSTANTS: List[Tuple[str, int]] = [
    ("pi", 0),
    ("e", 0),
]

# Assemble the full token list.  The concatenation order defines IDs.
_ALL_TOKENS: List[Tuple[str, int]] = (
    _SPECIAL
    + _BINARY_OPS
    + _UNARY_OPS
    + _FUNCTIONS
    + _INTEGRAL
    + _VARIABLES
    + _REAL_PARAMS
    + _INTEGER_PARAMS
    + _NUMERIC_INTEGERS
    + _NAMED_CONSTANTS
)


class Vocabulary:
    """Bidirectional mapping between tokens and integer IDs.

    The vocabulary is fixed at construction time.  Special tokens always
    occupy IDs 0-3 (PAD, BOS, EOS, SEP).

    >>> vocab = Vocabulary()
    >>> vocab.token_to_id("add")
    4
    >>> vocab.id_to_token(0)
    'PAD'
    >>> vocab.arity("sin")
    1
    >>> len(vocab)
    85
    """

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._token2id: Dict[str, int] = {}
        self._arities: Dict[str, int] = {}

        for idx, (token, arity) in enumerate(_ALL_TOKENS):
            if token in self._token2id:
                raise ValueError(
                    f"Duplicate token {token!r} at index {idx} "
                    f"(first seen at {self._token2id[token]})"
                )
            self._tokens.append(token)
            self._token2id[token] = idx
            self._arities[token] = arity

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def token_to_id(self, token: str) -> Optional[int]:
        """Return the integer ID for *token*, or ``None`` if unknown."""
        return self._token2id.get(token)

    def id_to_token(self, id: int) -> Optional[str]:
        """Return the token string for *id*, or ``None`` if out of range."""
        if 0 <= id < len(self._tokens):
            return self._tokens[id]
        return None

    def arity(self, token: str) -> int:
        """Return the number of children *token* expects in prefix notation.

        Raises ``KeyError`` if *token* is not in the vocabulary.
        """
        return self._arities[token]

    def __len__(self) -> int:
        """Return the total number of tokens in the vocabulary."""
        return len(self._tokens)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def __contains__(self, token: str) -> bool:
        return token in self._token2id

    def __repr__(self) -> str:
        return f"Vocabulary(size={len(self)})"

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def bos_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return 2

    @property
    def sep_id(self) -> int:
        return 3

    def tokens(self) -> List[str]:
        """Return a list of all tokens, ordered by ID."""
        return list(self._tokens)
