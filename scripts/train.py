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
from torch.utils.tensorboard import SummaryWriter

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


def _dataset_to_problems(ds: IntegrationDataset, vocab: Vocabulary) -> list:
    """Convert saved dataset samples back to Integral problems.

    Extracts the integrand tokens from the input_ids (between BOS and SEP),
    converts to SymPy, and wraps in ``Integral(f, x)``.
    """
    from integrate_zero.data.prefix import prefix_to_sympy

    problems = []
    for i in range(len(ds)):
        sample = ds[i]
        ids = sample["input_ids"].tolist()

        # Find BOS and SEP positions
        try:
            bos_pos = ids.index(vocab.bos_id)
            sep_pos = ids.index(vocab.sep_id)
        except ValueError:
            continue

        # Extract integrand token IDs (between BOS and SEP, exclusive)
        f_ids = ids[bos_pos + 1:sep_pos]
        f_tokens = [vocab.id_to_token(tid) for tid in f_ids]
        if None in f_tokens:
            continue

        try:
            f_expr = prefix_to_sympy(f_tokens)
            problems.append(sympy.Integral(f_expr, x))
        except Exception:
            continue

    return problems


def get_device() -> torch.device:
    """Select the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_model(
    vocab: Vocabulary,
    device: torch.device,
    d_model: int = 384,
    nhead: int = 6,
    num_layers: int = 8,
    d_ff: int = 1536,
    max_seq_len: int = 512,
) -> IntegrateZeroModel:
    """Create an IntegrateZeroModel."""
    # MPS has a known issue where dropout inside TransformerDecoder
    # combined with attention masks produces NaN.  Disable dropout on MPS.
    dropout = 0.0 if device.type == "mps" else 0.1
    return IntegrateZeroModel(
        vocab_size=len(vocab),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
    ).to(device)


# ---------------------------------------------------------------------------
# Phase entry points
# ---------------------------------------------------------------------------


def run_supervised(args: argparse.Namespace, model: IntegrateZeroModel,
                   vocab: Vocabulary, writer: SummaryWriter) -> None:
    """Phase 1: Generate or load dataset, train with SupervisedTrainer, save checkpoint."""
    data_dir = Path(args.data_dir) if args.data_dir else None
    val_dataset = None

    if data_dir and (data_dir / "train.pt").exists():
        print(f"Loading pre-generated training data from {data_dir / 'train.pt'}...")
        dataset = IntegrationDataset.load(data_dir / "train.pt")
        if (data_dir / "val.pt").exists():
            print(f"Loading validation data from {data_dir / 'val.pt'}...")
            val_dataset = IntegrationDataset.load(data_dir / "val.pt")
    else:
        print(f"Generating {args.num_samples} training pairs (max_depth={args.max_depth})...")
        dataset = IntegrationDataset(args.num_samples, max_depth=args.max_depth, seed=args.seed)

    print(f"Training dataset size: {len(dataset)}")
    if val_dataset is not None:
        print(f"Validation dataset size: {len(val_dataset)}")

    trainer = SupervisedTrainer(
        model, dataset, vocab,
        batch_size=args.batch_size, lr=args.lr,
        val_dataset=val_dataset,
    )

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        train_metrics = trainer.train_epoch()
        val_metrics = trainer.evaluate_loss()

        writer.add_scalar("train/total_loss", train_metrics["total_loss"], epoch)
        writer.add_scalar("train/policy_loss", train_metrics["policy_loss"], epoch)
        writer.add_scalar("train/value_loss", train_metrics["value_loss"], epoch)
        writer.add_scalar("val/total_loss", val_metrics["total_loss"], epoch)
        writer.add_scalar("val/policy_loss", val_metrics["policy_loss"], epoch)
        writer.add_scalar("val/value_loss", val_metrics["value_loss"], epoch)

        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train loss: {train_metrics['total_loss']:.4f} - "
            f"Eval loss: {val_metrics['total_loss']:.4f}"
        )

        # Save periodic checkpoint every 10 epochs and on best val loss
        val_loss = val_metrics["total_loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint("checkpoints/supervised_best.pt")
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(f"checkpoints/supervised_ep{epoch + 1}.pt")
            print(f"  Saved checkpoint: checkpoints/supervised_ep{epoch + 1}.pt")

    checkpoint_path = "checkpoints/supervised.pt"
    trainer.save_checkpoint(checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")


def run_rl(args: argparse.Namespace, model: IntegrateZeroModel,
           vocab: Vocabulary, writer: SummaryWriter) -> None:
    """Phase 2: Run RLTrainer iterations with MCTS, save checkpoint."""
    trainer = RLTrainer(
        model, vocab, lr=args.lr,
        num_candidates=args.num_candidates,
        search_budget=args.search_budget,
        max_steps=args.max_mcts_steps,
    )

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

        writer.add_scalar("rl/solve_rate", stats["solve_rate"], it)
        writer.add_scalar("rl/policy_loss", stats["policy_loss"], it)
        writer.add_scalar("rl/value_loss", stats["value_loss"], it)

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
             vocab: Vocabulary, writer: SummaryWriter) -> None:
    """Phase 3: Load or generate test problems, evaluate model and SymPy baseline."""
    evaluator = Evaluator(model, vocab)
    data_dir = Path(args.data_dir) if args.data_dir else None

    # Try loading pre-generated test problems
    test_sets = {}
    if data_dir:
        for name in ["test_in_dist", "test_ood", "test_textbook"]:
            path = data_dir / f"{name}.pt"
            if path.exists():
                test_sets[name] = path

    if test_sets:
        for name, path in test_sets.items():
            print(f"\n--- {name} (from {path}) ---")
            # Load dataset and generate integration problems from the stored pairs
            ds = IntegrationDataset.load(path)
            problems = _dataset_to_problems(ds, vocab)
            print(f"Problems loaded: {len(problems)}")
            if not problems:
                continue

            model_results = evaluator.evaluate(
                problems,
                search_budget=args.search_budget,
                num_candidates=args.num_candidates,
                max_steps=args.max_mcts_steps,
            )
            print(f"Model  - Solve rate: {model_results['solve_rate']:.3f}, "
                  f"Solved: {model_results['solved']}/{model_results['total']}")

            sympy_results = evaluator.sympy_baseline(problems)
            print(f"SymPy  - Solve rate: {sympy_results['solve_rate']:.3f}, "
                  f"Solved: {sympy_results['solved']}/{sympy_results['total']}")
    else:
        # Fallback: generate test problems on-the-fly
        print("Generating test problems...")
        test_problems = []
        for _ in range(args.num_eval_problems):
            try:
                f, _F = generate_training_pair(max_depth=args.max_depth)
                test_problems.append(sympy.Integral(f, x))
            except Exception:
                continue

        print(f"Test problems generated: {len(test_problems)}")

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
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible data generation",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None,
        help="Directory containing pre-generated datasets (train.pt, val.pt, etc.)",
    )
    parser.add_argument(
        "--num_candidates", type=int, default=8,
        help="MCTS candidates per expansion (default: 8)",
    )
    parser.add_argument(
        "--search_budget", type=int, default=200,
        help="MCTS search budget per problem (default: 200)",
    )
    parser.add_argument(
        "--max_mcts_steps", type=int, default=20,
        help="Max MCTS transformation steps (default: 20)",
    )
    parser.add_argument(
        "--d_model", type=int, default=384,
        help="Model embedding dimension (default: 384)",
    )
    parser.add_argument(
        "--nhead", type=int, default=6,
        help="Number of attention heads (default: 6)",
    )
    parser.add_argument(
        "--num_layers", type=int, default=8,
        help="Number of transformer layers (default: 8)",
    )
    parser.add_argument(
        "--d_ff", type=int, default=1536,
        help="Feed-forward dimension (default: 1536)",
    )
    parser.add_argument(
        "--max_seq_len", type=int, default=512,
        help="Max sequence length for position embeddings (default: 512)",
    )
    parser.add_argument(
        "--num_eval_problems", type=int, default=100,
        help="Number of test problems for eval phase (default: 100)",
    )
    parser.add_argument(
        "--log_dir", type=str, default="runs/",
        help="TensorBoard log directory (default: runs/)",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    vocab = Vocabulary()
    model = make_model(
        vocab, device,
        d_model=args.d_model, nhead=args.nhead,
        num_layers=args.num_layers, d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    Path("checkpoints").mkdir(exist_ok=True)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, weights_only=True, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: {args.checkpoint}")

    writer = SummaryWriter(log_dir=args.log_dir)
    print(f"TensorBoard log dir: {args.log_dir}")

    try:
        if args.phase == "supervised":
            run_supervised(args, model, vocab, writer)
        elif args.phase == "rl":
            run_rl(args, model, vocab, writer)
        elif args.phase == "eval":
            run_eval(args, model, vocab, writer)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
