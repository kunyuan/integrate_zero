# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IntegrateZero is a research project applying AlphaZero-style reinforcement learning to symbolic integration. The core insight: differentiation is trivial (perfect verifier), while integration is hard — so we can generate unlimited training data via the forward direction and learn the reverse.

**Status**: Planning/proposal stage — no implementation code yet. The repository contains two design documents.

## Planned Tech Stack

- **Language**: Python 3.8+
- **Deep Learning**: PyTorch (Transformer encoder-decoder, ~10-30M params)
- **Symbolic Math**: SymPy (expression generation, tokenization, derivative verification)
- **Search**: MCTS (Monte Carlo Tree Search)
- **Hardware**: Single GPU (RTX 3090/4090 or A100)

## Architecture (Primary Approach — IntegrateZero)

The system follows AlphaZero's architecture mapped to symbolic integration:

```
Expression (prefix notation tokens)
    → Transformer Encoder (4-6 layers, d=256-384)
    → Policy Head (Transformer Decoder) → candidate next expressions
    → Value Head (2-layer MLP) → P(solvable from this state)
    → MCTS explores candidates, derivative verification filters valid moves
    → Terminates when expression has no ∫ symbols
```

**State**: math expression (may contain ∫). **Action**: generate next expression autoregressively. **Reward**: +1 when no ∫ remains and derivative verification passes.

**Expression representation**: Prefix notation (Polish notation), vocabulary ~100-200 tokens (operators, functions, variables, constants, INT special token).

**Verification**: `sympy.diff(candidate, x)` checked against integrand via symbolic simplification, with numerical sampling fallback (20 random points).

## Training Pipeline

1. **Phase 1 — Supervised pretraining**: Generate random F(x), differentiate to get f(x), train on (f→F) pairs. Curriculum from depth 2 to 10+.
2. **Phase 2 — Self-play RL**: MCTS + policy/value networks, update from successful trajectories.
3. **Phase 3 — Auto-curriculum**: Adjust problem difficulty to maintain 30-80% success rate.

## Alternative Approach (hierarchical-rl-integration-project.md)

Uses hierarchical RL to learn rule selection over RUBI's 6,700 hand-crafted integration rules, with Wolfram Engine as execution backend. Kept as reference — the primary approach (IntegrateZero) avoids external rule dependencies entirely.

## Key Design Decisions

- **No external rule library**: Rules are learned implicitly in model weights, not from RUBI or any CAS
- **Pure SymPy + PyTorch**: No dependency on Wolfram Engine or commercial CAS after training
- **Multi-step reasoning**: Unlike Lample & Charton's one-shot seq2seq, this uses iterative expression transformation with per-step verification
- **SymPy RUBI port failure is a cautionary tale**: SymPy spent 8 years trying to port RUBI's 6,700 rules and deleted 87,806 lines. This project sidesteps that entirely by learning rules implicitly.

## Evaluation

- **Benchmarks**: RUBI test set (72,944 problems), Lample & Charton FWD/BWD (~5K each), self-generated OOD (~10K), textbook problems (~500)
- **Baselines**: RUBI 4.16.1, Mathematica, SymPy, Lample & Charton seq2seq, AlphaIntegrator
- **Metrics**: Solve rate, average steps, OOD generalization, search efficiency

## Key References

- AlphaZero (Silver et al., 2018)
- AlphaIntegrator (Unsal, Gehr & Vechev, 2024) — arXiv:2410.02666
- Lample & Charton (2019) — arXiv:1912.01412
- RUBI — rulebasedintegration.org
