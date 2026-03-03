"""IntegrateZero training script.

Usage:
    python scripts/train.py --phase supervised --num_samples 10000 --epochs 10
    python scripts/train.py --phase rl --checkpoint checkpoints/supervised.pt --iterations 50
    python scripts/train.py --phase eval --checkpoint checkpoints/rl.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sympy
import torch

from integrate_zero.data.dataset import IntegrationDataset
from integrate_zero.data.generator import generate_training_pair
from integrate_zero.data.vocabulary import Vocabulary
from integrate_zero.eval.evaluate import Evaluator
from integrate_zero.model.transformer import IntegrateZeroModel
from integrate_zero.train.rl import RLTrainer
from integrate_zero.train.supervised import SupervisedTrainer

x = sympy.Symbol("x")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    """Select the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_model(vocab: Vocabulary, device: torch.device) -> IntegrateZeroModel:
    """Create an IntegrateZeroModel with full-size defaults."""
    return IntegrateZeroModel(
        vocab_size=len(vocab),
        d_model=384,
        nhead=6,
        num_layers=8,
        d_ff=1536,
    ).to(device)


# ---------------------------------------------------------------------------
# Phase entry points
# ---------------------------------------------------------------------------


def run_supervised(args: argparse.Namespace, model: IntegrateZeroModel,
                   vocab: Vocabulary) -> None:
    """Phase 1: Generate dataset, train with SupervisedTrainer, save checkpoint."""
    print(f"Generating {args.num_samples} training pairs (max_depth={args.max_depth})...")
    dataset = IntegrationDataset(args.num_samples, max_depth=args.max_depth)
    print(f"Dataset size: {len(dataset)}")

    trainer = SupervisedTrainer(
        model, dataset, vocab,
        batch_size=args.batch_size, lr=args.lr,
    )

    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch()
        eval_loss = trainer.evaluate_loss()
        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train loss: {train_loss:.4f} - Eval loss: {eval_loss:.4f}"
        )

    checkpoint_path = "checkpoints/supervised.pt"
    trainer.save_checkpoint(checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


def run_rl(args: argparse.Namespace, model: IntegrateZeroModel,
           vocab: Vocabulary) -> None:
    """Phase 2: Run RLTrainer iterations with MCTS, save checkpoint."""
    trainer = RLTrainer(model, vocab, lr=args.lr)

    for it in range(args.iterations):
        # Generate fresh problems each iteration
        problems = []
        for _ in range(args.problems_per_iter):
            try:
                f, _F = generate_training_pair(max_depth=args.max_depth)
                problems.append(sympy.Integral(f, x))
            except Exception:
                continue

        stats = trainer.train_iteration(problems)
        print(
            f"Iteration {it + 1}/{args.iterations} - "
            f"Solve rate: {stats['solve_rate']:.3f} - "
            f"Policy loss: {stats['policy_loss']:.4f} - "
            f"Value loss: {stats['value_loss']:.4f}"
        )

    checkpoint_path = "checkpoints/rl.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


def run_eval(args: argparse.Namespace, model: IntegrateZeroModel,
             vocab: Vocabulary) -> None:
    """Phase 3: Generate test problems, evaluate model and SymPy baseline."""
    print("Generating test problems...")
    test_problems = []
    for _ in range(100):
        try:
            f, _F = generate_training_pair(max_depth=args.max_depth)
            test_problems.append(sympy.Integral(f, x))
        except Exception:
            continue

    print(f"Test problems generated: {len(test_problems)}")

    evaluator = Evaluator(model, vocab)

    print("\n--- Model Evaluation ---")
    model_results = evaluator.evaluate(test_problems)
    print(f"Solve rate: {model_results['solve_rate']:.3f}")
    print(f"Avg steps:  {model_results['avg_steps']:.1f}")
    print(f"Solved:     {model_results['solved']}/{model_results['total']}")

    print("\n--- SymPy Baseline ---")
    sympy_results = evaluator.sympy_baseline(test_problems)
    print(f"Solve rate: {sympy_results['solve_rate']:.3f}")
    print(f"Solved:     {sympy_results['solved']}/{sympy_results['total']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IntegrateZero training script",
    )
    parser.add_argument(
        "--phase", choices=["supervised", "rl", "eval"], required=True,
        help="Training phase to run",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to load checkpoint from",
    )
    parser.add_argument(
        "--num_samples", type=int, default=100000,
        help="Number of training samples for supervised phase (default: 100000)",
    )
    parser.add_argument(
        "--max_depth", type=int, default=5,
        help="Max expression depth (default: 5)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Supervised training epochs (default: 10)",
    )
    parser.add_argument(
        "--iterations", type=int, default=50,
        help="RL iteration rounds (default: 50)",
    )
    parser.add_argument(
        "--problems_per_iter", type=int, default=2000,
        help="Problems per RL iteration (default: 2000)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Batch size (default: 256)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--pretrain", action=argparse.BooleanOptionalAction, default=True,
        help="Whether to use supervised pretraining (default: True, for future ablation)",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    vocab = Vocabulary()
    model = make_model(vocab, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    Path("checkpoints").mkdir(exist_ok=True)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, weights_only=True, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")

    if args.phase == "supervised":
        run_supervised(args, model, vocab)
    elif args.phase == "rl":
        run_rl(args, model, vocab)
    elif args.phase == "eval":
        run_eval(args, model, vocab)


if __name__ == "__main__":
    main()
