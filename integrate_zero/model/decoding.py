"""Arity-constrained decoding for prefix-notation expression generation.

During autoregressive generation the model must produce syntactically valid
prefix expressions.  The :class:`ArityMask` tracks remaining open slots in
the expression tree and restricts the set of tokens the model may emit at
each step, guaranteeing that:

* EOS is only produced when the expression is complete (remaining slots == 0).
* EOS is blocked while the expression is still incomplete.
* Special tokens (PAD, BOS, SEP) are never emitted during generation.
"""

from __future__ import annotations

from typing import Set, List

from integrate_zero.data.vocabulary import Vocabulary


# Special tokens that are never allowed during generation.
_BLOCKED_SPECIALS = frozenset({"PAD", "BOS", "SEP"})


class ArityMask:
    """Enforce syntactically valid prefix expression generation.

    The mask uses an *arity stack* counter:

    * Start with ``remaining = 1`` (we need one root expression).
    * For each generated token: ``remaining -= 1`` (fills a slot),
      ``remaining += arity(token)`` (opens new child slots).
    * ``remaining == 0`` means the expression is complete --> force EOS.
    * ``remaining > 0`` means the expression is incomplete --> block EOS.

    Parameters
    ----------
    vocab : Vocabulary
        The shared vocabulary instance.
    """

    def __init__(self, vocab: Vocabulary) -> None:
        self._vocab = vocab

        # Pre-compute the set of "content" token IDs (everything except
        # the blocked specials and EOS).
        self._content_ids: Set[int] = set()
        self._eos_id: int = vocab.token_to_id("EOS")

        for token in vocab.tokens():
            if token in _BLOCKED_SPECIALS or token == "EOS":
                continue
            tid = vocab.token_to_id(token)
            if tid is not None:
                self._content_ids.add(tid)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_allowed_tokens(self, generated_tokens: List[str]) -> Set[int]:
        """Return the set of allowed token IDs given tokens generated so far.

        Parameters
        ----------
        generated_tokens : list[str]
            Tokens generated so far *after* the SEP token (i.e. only the
            output-side tokens that form the antiderivative expression).

        Returns
        -------
        set[int]
            Token IDs the model is allowed to emit next.
        """
        remaining = self._compute_remaining(generated_tokens)

        if remaining == 0:
            # Expression is complete -- only EOS is valid.
            return {self._eos_id}
        else:
            # Expression is incomplete -- allow all content tokens, block EOS.
            return set(self._content_ids)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_remaining(self, tokens: List[str]) -> int:
        """Compute the number of remaining open slots in the expression.

        Starting from 1 (need one root expression), for each token we
        subtract 1 (fills a slot) and add its arity (opens child slots).
        """
        remaining = 1
        for token in tokens:
            remaining -= 1
            remaining += self._vocab.arity(token)
        return remaining
