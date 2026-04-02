# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IntegrateZero is a research project applying AlphaZero-style reinforcement learning to symbolic integration. The core insight: differentiation is trivial (perfect verifier), while integration is hard — so we can generate unlimited training data via the forward direction and learn the reverse.

## Build & Run Commands

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run all tests
pytest

# Run a specific test file
pytest tests/test_prefix.py -v

# Run a specific test
pytest tests/test_prefix.py::test_roundtrip_complex -v

# Training: supervised pretraining (Phase 1)
python scripts/train.py --phase supervised --num_samples 10000 --epochs 10

# Training: RL self-play (Phase 2)
python scripts/train.py --phase rl --checkpoint checkpoints/supervised.pt --iterations 50

# Evaluation
python scripts/train.py --phase eval --checkpoint checkpoints/rl.pt

# Quick smoke test (tiny params, CPU-friendly)
python scripts/train.py --phase supervised --num_samples 50 --epochs 1 --batch_size 8
```

## Architecture

**Model**: Decoder-only Transformer (GPT-style), 8 layers, d_model=384, ~12-19M params. Takes `[BOS] A_tokens [SEP] B_tokens [EOS]` as input. Policy head predicts next tokens after SEP; value head extracts from SEP position → MLP → P(solvable).

**Multi-step transformation**: The model iteratively transforms expressions containing `∫` symbols. Each step either:
- **Integration step**: reduces/eliminates `∫` — verified by `d/dx(B) == integrand(A)`
- **Identity rewrite**: algebraically equivalent — verified by `simplify(A - B) == 0`

Terminates when no `∫` remains (reward +1) or max steps exceeded (failure).

**Expression representation**: Prefix notation (Polish notation), ~85 token vocabulary. `INT` token has arity=2 (integrand + variable). Arity-constrained decoding guarantees syntactically valid expressions.

**Training pipeline**:
1. Phase 1 — Supervised: Generate random F(x), differentiate to get f(x), train seq2seq on (f→F)
2. Phase 2 — RL: MCTS explores multi-step solutions, train from successful trajectories
3. `--pretrain` / `--no-pretrain` flag controls whether Phase 1 is used (ablation experiment)

## Code Structure

```
integrate_zero/
  data/
    vocabulary.py     # ~85 token vocab with arity tracking
    prefix.py         # Bidirectional prefix notation <-> SymPy conversion
    generator.py      # Random expression generation, (f, F) training pairs
    dataset.py        # PyTorch Dataset with tokenization and collation
  model/
    transformer.py    # Decoder-only Transformer + value head
    decoding.py       # ArityMask for constrained prefix generation
  mcts/
    search.py         # MCTSNode + MCTS with UCB selection
  train/
    supervised.py     # Phase 1: supervised pretraining
    rl.py             # Phase 2: RL with MCTS episode collection
  verify/
    verifier.py       # Step verification (integration/identity/invalid)
  eval/
    evaluate.py       # Model evaluation + SymPy baseline
scripts/
  train.py            # Main CLI training script
```

## Key Design Decisions

- **General position assumption**: Symbolic parameters assume non-degenerate values. Safe because verification only differentiates (forward direction).
- **Two step types**: Integration steps (reduce `∫`) and identity rewrites (algebraically equivalent). Reward: only +1 at terminal; identity steps get 0.
- **Arity-constrained decoding**: Maintains an arity stack during autoregressive generation. Forces EOS when expression is complete, blocks EOS when incomplete.
- **Integer vs real parameters**: `k`, `l`, `m`, `n` are `Symbol(integer=True)`; `a`, `b`, `c`, `d` are `Symbol(real=True)`.
- **Device-agnostic**: Supports CUDA, MPS (MacBook), and CPU.

## Key References

- AlphaZero (Silver et al., 2018)
- AlphaIntegrator (Unsal, Gehr & Vechev, 2024) — arXiv:2410.02666
- Lample & Charton (2019) — arXiv:1912.01412
- RUBI — rulebasedintegration.org

## Design Documents

- `docs/plans/2026-03-03-integrate-zero-demo-design.md` — Full design specification
- `docs/plans/2026-03-03-integrate-zero-impl-plan.md` — Implementation plan with 14 tasks
- `integrate_zero_proposal.md` — Original research proposal (Chinese)
- `hierarchical-rl-integration-project.md` — Alternative approach proposal (Chinese)
