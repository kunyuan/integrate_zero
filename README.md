# IntegrateZero

AlphaZero-style reinforcement learning for symbolic integration.

**Core insight**: differentiation is trivial (perfect verifier), while integration is hard. We generate unlimited training data by differentiating random expressions, then learn to integrate via self-play.

## How It Works

The model learns multi-step symbolic integration through two phases:

1. **Supervised pretraining** -- Generate random F(x), differentiate to get f(x), train on (f -> F) pairs
2. **RL self-play** -- MCTS explores multi-step integration paths, train from successful trajectories

Each step is either an **integration step** (reduces/eliminates integrals, verified by differentiation) or an **identity rewrite** (algebraically equivalent, verified by simplification). The process terminates when no integral symbols remain.

### Model

Decoder-only Transformer (GPT-style), 8 layers, d_model=384, ~12-19M parameters. Dual-headed: policy head for next-token prediction, value head for estimating P(solvable). Expressions are represented in prefix notation (~85 token vocabulary) with arity-constrained decoding to guarantee syntactic validity.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest

# Quick smoke test (CPU-friendly, ~1 min)
python scripts/train.py --phase supervised --num_samples 50 --epochs 1 --batch_size 8
```

## Training

```bash
# Phase 1: Supervised pretraining
python scripts/train.py --phase supervised --num_samples 10000 --epochs 10

# Phase 2: RL self-play
python scripts/train.py --phase rl --checkpoint checkpoints/supervised.pt --iterations 50

# Evaluation
python scripts/train.py --phase eval --checkpoint checkpoints/rl.pt
```

Supports CUDA, MPS (Apple Silicon), and CPU.

## Project Structure

```
integrate_zero/
  data/
    vocabulary.py     # ~85 token vocab with arity tracking
    prefix.py         # Prefix notation <-> SymPy conversion
    generator.py      # Random expression & training pair generation
    dataset.py        # PyTorch Dataset with tokenization
  model/
    transformer.py    # Decoder-only Transformer + value head
    decoding.py       # Arity-constrained decoding
  mcts/
    search.py         # MCTS with UCB selection
  train/
    supervised.py     # Phase 1: supervised pretraining
    rl.py             # Phase 2: RL with MCTS episodes
  verify/
    verifier.py       # Step verification (integration/identity/invalid)
  eval/
    evaluate.py       # Benchmarking + SymPy baseline comparison
scripts/
  train.py            # Main CLI entry point
```

## References

- Silver et al. (2018) -- AlphaZero
- Unsal, Gehr & Vechev (2024) -- [AlphaIntegrator](https://arxiv.org/abs/2410.02666)
- Lample & Charton (2019) -- [Deep Learning for Symbolic Mathematics](https://arxiv.org/abs/1912.01412)
- [RUBI](https://rulebasedintegration.org) -- Rule-Based Integration

## License

MIT
