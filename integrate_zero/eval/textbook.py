"""Hand-crafted textbook integration problems for evaluation.

Provides ~30 classic integrals across several categories:
polynomials, trigonometric, exponential, integration by parts,
and substitution.

Each problem is an ``(Integral, expected_antiderivative)`` pair.
The expected antiderivative is given for reference — during evaluation
the verifier checks correctness, so the exact form doesn't matter.

Usage::

    from integrate_zero.eval.textbook import TEXTBOOK_PROBLEMS, get_textbook_dataset

    # List of (Integral, F_expected) pairs
    for integral, F in TEXTBOOK_PROBLEMS:
        print(integral, "->", F)

    # As an IntegrationDataset (for consistency with training pipeline)
    ds = get_textbook_dataset()
"""

from __future__ import annotations

from typing import List, Tuple

import sympy
from sympy import (
    Integral,
    Rational,
    Symbol,
    cos,
    exp,
    log,
    sin,
    sqrt,
    tan,
    atan,
    sinh,
    cosh,
    tanh,
)

x = Symbol("x")

# ---------------------------------------------------------------------------
# Textbook integration problems: (Integral, expected_F)
#
# Categories:
#   1. Polynomials
#   2. Trigonometric
#   3. Exponential / logarithmic
#   4. Integration by parts
#   5. Substitution
#   6. Hyperbolic
# ---------------------------------------------------------------------------

TEXTBOOK_PROBLEMS: List[Tuple[sympy.Expr, sympy.Expr]] = [
    # --- 1. Polynomials ---
    (Integral(x**2, x), x**3 / 3),
    (Integral(x**3, x), x**4 / 4),
    (Integral(x**2 + 2*x + 1, x), x**3 / 3 + x**2 + x),
    (Integral(3*x**2 - 2*x + 5, x), x**3 - x**2 + 5*x),
    (Integral(x**5, x), x**6 / 6),
    (Integral(1 + x, x), x + x**2 / 2),

    # --- 2. Trigonometric ---
    (Integral(sin(x), x), -cos(x)),
    (Integral(cos(x), x), sin(x)),
    (Integral(sin(x)**2, x), x/2 - sin(x)*cos(x)/2),
    (Integral(cos(x)**2, x), x/2 + sin(x)*cos(x)/2),
    (Integral(sin(2*x), x), -cos(2*x) / 2),
    (Integral(cos(3*x), x), sin(3*x) / 3),

    # --- 3. Exponential / logarithmic ---
    (Integral(exp(x), x), exp(x)),
    (Integral(exp(2*x), x), exp(2*x) / 2),
    (Integral(x * exp(x), x), x * exp(x) - exp(x)),
    (Integral(exp(-x), x), -exp(-x)),

    # --- 4. Integration by parts ---
    (Integral(x * cos(x), x), x * sin(x) + cos(x)),
    (Integral(x * sin(x), x), -x * cos(x) + sin(x)),
    (Integral(x**2 * exp(x), x), x**2 * exp(x) - 2*x*exp(x) + 2*exp(x)),

    # --- 5. Substitution ---
    (Integral(2*x * cos(x**2), x), sin(x**2)),
    (Integral(cos(x) * exp(sin(x)), x), exp(sin(x))),
    (Integral(2*x * exp(x**2), x), exp(x**2)),
    (Integral(3*x**2 * cos(x**3), x), sin(x**3)),

    # --- 6. Hyperbolic ---
    (Integral(sinh(x), x), cosh(x)),
    (Integral(cosh(x), x), sinh(x)),
    (Integral(x * cosh(x), x), x * sinh(x) - cosh(x)),

    # --- 7. Simple powers and roots ---
    (Integral(x**Rational(1, 2), x), Rational(2, 3) * x**Rational(3, 2)),
    (Integral(x**(-1), x), log(x)),

    # --- 8. Composite ---
    (Integral(sin(x) + cos(x), x), -cos(x) + sin(x)),
    (Integral(exp(x) + x, x), exp(x) + x**2 / 2),
]


def get_textbook_problems() -> List[sympy.Expr]:
    """Return just the Integral expressions (for use with Evaluator)."""
    return [problem for problem, _expected in TEXTBOOK_PROBLEMS]


def get_textbook_dataset():
    """Return textbook problems as an IntegrationDataset.

    Converts each ``(Integral(f, x), F)`` into the dataset format
    ``(f, F)`` where ``f`` is the integrand and ``F`` is the
    antiderivative. Problems that fail tokenization are skipped.

    Returns
    -------
    IntegrationDataset
        A dataset containing the textbook problems.
    """
    from integrate_zero.data.dataset import IntegrationDataset
    from integrate_zero.data.prefix import sympy_to_prefix
    from integrate_zero.data.vocabulary import Vocabulary
    from torch.utils.data import Dataset

    import torch

    vocab = Vocabulary()
    ds = IntegrationDataset.__new__(IntegrationDataset)
    Dataset.__init__(ds)
    ds.vocab = vocab
    ds.max_len = 512
    ds.samples = []

    for integral, F in TEXTBOOK_PROBLEMS:
        # Extract the integrand from the Integral
        f = integral.args[0]
        try:
            f_tokens = sympy_to_prefix(f)
            F_tokens = sympy_to_prefix(F)
        except (ValueError, TypeError, AttributeError):
            continue

        # Check all tokens are in vocabulary
        if not all(vocab.token_to_id(t) is not None for t in f_tokens):
            continue
        if not all(vocab.token_to_id(t) is not None for t in F_tokens):
            continue

        # Build the sample
        f_ids = [vocab.token_to_id(t) for t in f_tokens]
        F_ids = [vocab.token_to_id(t) for t in F_tokens]
        input_ids = [vocab.bos_id] + f_ids + [vocab.sep_id] + F_ids + [vocab.eos_id]

        if len(input_ids) > ds.max_len:
            continue

        num_masked = 1 + len(f_ids) + 1  # BOS + f_tokens + SEP
        target_ids = [-100] * num_masked + F_ids + [vocab.eos_id]

        ds.samples.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "value_label": 1.0,
        })

    return ds
