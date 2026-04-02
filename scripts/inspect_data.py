"""Inspect data quality for IntegrateZero training data.

Generates a batch of training pairs and reports statistics including
success rate, token lengths, expression depths, operator frequencies,
sample pairs, and prefix roundtrip verification.

Usage:
    python scripts/inspect_data.py
    python scripts/inspect_data.py --num_samples 500 --seed 42
    python scripts/inspect_data.py --load data/train.pt
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import sympy
from sympy import Symbol

from integrate_zero.data.dataset import IntegrationDataset
from integrate_zero.data.generator import generate_training_pair
from integrate_zero.data.prefix import sympy_to_prefix, prefix_to_sympy
from integrate_zero.data.vocabulary import Vocabulary

x = Symbol("x")


def _expr_depth(expr: sympy.Expr) -> int:
    """Compute the depth of a SymPy expression tree."""
    if not expr.args:
        return 0
    return 1 + max(_expr_depth(a) for a in expr.args)


def _count_operators(tokens: list[str], counter: Counter) -> None:
    """Count operator/function tokens in a prefix token list."""
    vocab = Vocabulary()
    for tok in tokens:
        if vocab.token_to_id(tok) is not None and vocab.arity(tok) > 0:
            counter[tok] += 1


def inspect_generated(num_samples: int, max_depth: int, seed: int | None) -> None:
    """Generate pairs on-the-fly and report statistics."""
    if seed is not None:
        random.seed(seed)

    pairs = []
    attempts = 0
    max_attempts = num_samples * 5

    while len(pairs) < num_samples and attempts < max_attempts:
        attempts += 1
        try:
            f, F = generate_training_pair(max_depth=max_depth)
            pairs.append((f, F))
        except Exception:
            continue

    _report(pairs, attempts, num_samples)


def inspect_saved(path: str) -> None:
    """Load a saved dataset and report statistics."""
    ds = IntegrationDataset.load(path)
    print(f"Loaded dataset from {path}")
    print(f"  Samples: {len(ds)}")

    # Extract token length statistics from saved samples
    lengths = []
    for i in range(len(ds)):
        sample = ds[i]
        # Count non-padding tokens
        ids = sample["input_ids"]
        length = (ids != 0).sum().item()
        lengths.append(length)

    if lengths:
        print(f"\nToken sequence lengths (including BOS/SEP/EOS):")
        print(f"  Min:  {min(lengths)}")
        print(f"  Mean: {sum(lengths) / len(lengths):.1f}")
        print(f"  Max:  {max(lengths)}")

    print(f"\nTo inspect expression-level details, use --num_samples to generate fresh data.")


def _report(pairs: list, attempts: int, requested: int) -> None:
    """Print a full quality report for generated pairs."""
    vocab = Vocabulary()

    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)

    # Success rate
    print(f"\nGeneration:")
    print(f"  Requested:  {requested}")
    print(f"  Generated:  {len(pairs)}")
    print(f"  Attempts:   {attempts}")
    print(f"  Success rate: {len(pairs) / max(attempts, 1) * 100:.1f}%")

    if not pairs:
        print("\nNo pairs generated — nothing to report.")
        return

    # Token sequence lengths
    f_lengths = []
    F_lengths = []
    total_lengths = []
    f_depths = []
    F_depths = []
    op_counter = Counter()
    roundtrip_ok = 0

    for f, F in pairs:
        try:
            f_tok = sympy_to_prefix(f)
            F_tok = sympy_to_prefix(F)
        except Exception:
            continue

        f_lengths.append(len(f_tok))
        F_lengths.append(len(F_tok))
        # Full sequence: [BOS] f_tok [SEP] F_tok [EOS]
        total_lengths.append(1 + len(f_tok) + 1 + len(F_tok) + 1)

        f_depths.append(_expr_depth(f))
        F_depths.append(_expr_depth(F))

        _count_operators(f_tok, op_counter)
        _count_operators(F_tok, op_counter)

        # Roundtrip verification
        try:
            f_rt = prefix_to_sympy(f_tok)
            F_rt = prefix_to_sympy(F_tok)
            if f_rt is not None and F_rt is not None:
                roundtrip_ok += 1
        except Exception:
            pass

    print(f"\nToken sequence lengths (f only):")
    print(f"  Min:  {min(f_lengths)}")
    print(f"  Mean: {sum(f_lengths) / len(f_lengths):.1f}")
    print(f"  Max:  {max(f_lengths)}")

    print(f"\nToken sequence lengths (F only):")
    print(f"  Min:  {min(F_lengths)}")
    print(f"  Mean: {sum(F_lengths) / len(F_lengths):.1f}")
    print(f"  Max:  {max(F_lengths)}")

    print(f"\nFull sequence lengths [BOS] f [SEP] F [EOS]:")
    print(f"  Min:  {min(total_lengths)}")
    print(f"  Mean: {sum(total_lengths) / len(total_lengths):.1f}")
    print(f"  Max:  {max(total_lengths)}")

    print(f"\nExpression depth (f):")
    print(f"  Min:  {min(f_depths)}")
    print(f"  Mean: {sum(f_depths) / len(f_depths):.1f}")
    print(f"  Max:  {max(f_depths)}")

    print(f"\nExpression depth (F):")
    print(f"  Min:  {min(F_depths)}")
    print(f"  Mean: {sum(F_depths) / len(F_depths):.1f}")
    print(f"  Max:  {max(F_depths)}")

    # Operator/function frequency
    print(f"\nOperator / function frequency (across f and F):")
    for tok, count in op_counter.most_common():
        print(f"  {tok:10s}: {count}")

    # Prefix roundtrip verification
    print(f"\nPrefix roundtrip verification: {roundtrip_ok}/{len(pairs)} OK "
          f"({roundtrip_ok / len(pairs) * 100:.1f}%)")

    # Sample 10 random (f, F) pairs
    print(f"\n{'=' * 60}")
    print("SAMPLE PAIRS (10 random)")
    print("=" * 60)
    sample_indices = random.sample(range(len(pairs)), min(10, len(pairs)))
    for idx in sample_indices:
        f, F = pairs[idx]
        print(f"\n  f = {f}")
        print(f"  F = {F}")
        print(f"  Verify: diff(F, x) = {sympy.diff(F, x)}")
        try:
            f_tok = sympy_to_prefix(f)
            print(f"  Prefix(f) = {' '.join(f_tok)}")
        except Exception:
            print(f"  Prefix(f) = <conversion failed>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect IntegrateZero data quality")
    parser.add_argument(
        "--num_samples", type=int, default=200,
        help="Number of samples to generate (default: 200)",
    )
    parser.add_argument(
        "--max_depth", type=int, default=5,
        help="Max expression depth (default: 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--load", type=str, default=None,
        help="Path to a saved .pt dataset file to inspect instead of generating",
    )
    args = parser.parse_args()

    if args.load:
        inspect_saved(args.load)
    else:
        inspect_generated(args.num_samples, args.max_depth, args.seed)


if __name__ == "__main__":
    main()
