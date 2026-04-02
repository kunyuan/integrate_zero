"""Pre-generate and save datasets for IntegrateZero training.

Usage:
    python scripts/generate_data.py --seed 42
    python scripts/generate_data.py --seed 42 --train_size 50000 --out_dir data
"""

from __future__ import annotations

import argparse
from pathlib import Path

from integrate_zero.data.dataset import IntegrationDataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and save IntegrateZero datasets",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--out_dir", type=str, default="data",
        help="Output directory for saved datasets (default: data)",
    )
    parser.add_argument(
        "--train_size", type=int, default=100000,
        help="Number of training samples (default: 100000)",
    )
    parser.add_argument(
        "--val_size", type=int, default=5000,
        help="Number of validation samples (default: 5000)",
    )
    parser.add_argument(
        "--test_size", type=int, default=5000,
        help="Number of in-distribution test samples (default: 5000)",
    )
    parser.add_argument(
        "--ood_size", type=int, default=2000,
        help="Number of out-of-distribution test samples (default: 2000)",
    )
    parser.add_argument(
        "--max_depth", type=int, default=5,
        help="Max expression depth for train/val/test (default: 5)",
    )
    parser.add_argument(
        "--ood_depth", type=int, default=8,
        help="Max expression depth for OOD test set (default: 8)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed

    # Training set
    print(f"Generating training set ({args.train_size} samples, seed={seed})...")
    train_ds = IntegrationDataset(
        args.train_size, max_depth=args.max_depth, seed=seed,
    )
    train_ds.save(out_dir / "train.pt")
    print(f"  Saved {len(train_ds)} samples to {out_dir / 'train.pt'}")

    # Validation set (different seed)
    val_seed = seed + 1
    print(f"Generating validation set ({args.val_size} samples, seed={val_seed})...")
    val_ds = IntegrationDataset(
        args.val_size, max_depth=args.max_depth, seed=val_seed,
    )
    val_ds.save(out_dir / "val.pt")
    print(f"  Saved {len(val_ds)} samples to {out_dir / 'val.pt'}")

    # In-distribution test set (different seed)
    test_seed = seed + 2
    print(f"Generating in-distribution test set ({args.test_size} samples, seed={test_seed})...")
    test_ds = IntegrationDataset(
        args.test_size, max_depth=args.max_depth, seed=test_seed,
    )
    test_ds.save(out_dir / "test_in_dist.pt")
    print(f"  Saved {len(test_ds)} samples to {out_dir / 'test_in_dist.pt'}")

    # Out-of-distribution test set (deeper expressions, different seed)
    ood_seed = seed + 3
    print(f"Generating OOD test set ({args.ood_size} samples, depth={args.ood_depth}, seed={ood_seed})...")
    ood_ds = IntegrationDataset(
        args.ood_size, max_depth=args.ood_depth, seed=ood_seed,
    )
    ood_ds.save(out_dir / "test_ood.pt")
    print(f"  Saved {len(ood_ds)} samples to {out_dir / 'test_ood.pt'}")

    # Textbook test set
    try:
        from integrate_zero.eval.textbook import get_textbook_dataset
        textbook_ds = get_textbook_dataset()
        textbook_ds.save(out_dir / "test_textbook.pt")
        print(f"  Saved {len(textbook_ds)} textbook problems to {out_dir / 'test_textbook.pt'}")
    except ImportError:
        print("  Skipping textbook set (integrate_zero.eval.textbook not found)")

    print("\nDone! Generated datasets:")
    for f in sorted(out_dir.glob("*.pt")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
