"""Reinforcement learning training loop for IntegrateZero (Phase 2).

Uses MCTS for episode collection and trains the model on successful
trajectories.  Each iteration:

1. Runs MCTS ``search`` on a batch of integration problems.
2. Filters for solved episodes (trajectories where the expression
   becomes terminal).
3. For each solved trajectory, iterates over consecutive
   ``(state, next_state)`` pairs.
4. Trains the policy head to predict the next state given the current
   state, and the value head to predict the reward signal.

The training loss mirrors the supervised loop:

- **Policy loss**: ``CrossEntropyLoss(ignore_index=-100)`` on logits
  after the SEP token.
- **Value loss**: ``BCELoss`` on the value head output.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import torch
import torch.nn as nn

from integrate_zero.data.prefix import sympy_to_prefix
from integrate_zero.data.vocabulary import Vocabulary
from integrate_zero.mcts.search import MCTS
from integrate_zero.model.transformer import IntegrateZeroModel


class RLTrainer:
    """RL trainer that collects MCTS episodes and trains on successful ones.

    Parameters
    ----------
    model : IntegrateZeroModel
        The transformer model with policy and value heads.
    vocab : Vocabulary
        Vocabulary for encoding/decoding expressions.
    num_candidates : int
        Number of candidate expressions generated per MCTS expansion.
    search_budget : int
        Total MCTS iterations per search call.
    max_steps : int
        Maximum number of sequential steps in MCTS.
    c_puct : float
        Exploration constant for the UCB formula.
    lr : float
        Learning rate for the Adam optimizer.
    """

    def __init__(
        self,
        model: IntegrateZeroModel,
        vocab: Vocabulary,
        num_candidates: int = 8,
        search_budget: int = 200,
        max_steps: int = 20,
        c_puct: float = 1.5,
        lr: float = 1e-4,
    ) -> None:
        self.model = model
        self.vocab = vocab
        self.device = next(model.parameters()).device

        self.mcts = MCTS(
            model=model,
            vocab=vocab,
            num_candidates=num_candidates,
            search_budget=search_budget,
            max_steps=max_steps,
            c_puct=c_puct,
        )

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.policy_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.value_loss_fn = nn.BCELoss()

        self.sep_id = vocab.sep_id

    def collect_episode(self, problem) -> dict:
        """Run MCTS on a problem, return trajectory data.

        Parameters
        ----------
        problem : sympy.Expr
            An integration problem (typically an ``Integral``).

        Returns
        -------
        dict
            ``{"problem": problem, "trajectory": [...], "solved": bool}``
            where ``trajectory`` is a list of SymPy expressions from root
            to solution (or partial trajectory if unsolved).
        """
        trajectory = self.mcts.search(problem)

        if trajectory is not None:
            return {
                "problem": problem,
                "trajectory": trajectory,
                "solved": True,
            }
        else:
            return {
                "problem": problem,
                "trajectory": [],
                "solved": False,
            }

    def train_iteration(self, problems: list) -> dict:
        """Run one iteration: collect episodes on all problems, then update.

        For each problem, runs MCTS to collect an episode.  Then trains
        on all solved episodes by iterating over ``(state, next_state)``
        pairs in the trajectory.

        Parameters
        ----------
        problems : list[sympy.Expr]
            List of integration problems to attempt.

        Returns
        -------
        dict
            ``{"policy_loss": float, "value_loss": float, "solve_rate": float}``
        """
        # Collect episodes in parallel using threads.
        # GPU inference is thread-safe; SymPy verification uses ProcessPool.
        self.model.eval()
        with ThreadPoolExecutor(max_workers=min(len(problems), 16)) as executor:
            episodes = list(executor.map(self.collect_episode, problems))

        solved_count = sum(1 for ep in episodes if ep["solved"])
        solve_rate = solved_count / len(problems) if problems else 0.0

        # Train on solved episodes
        policy_losses: List[float] = []
        value_losses: List[float] = []

        self.model.train()

        for episode in episodes:
            if not episode["solved"]:
                continue
            trajectory = episode["trajectory"]
            # Iterate over consecutive (state, next_state) pairs
            for i in range(len(trajectory) - 1):
                state = trajectory[i]
                next_state = trajectory[i + 1]
                result = self._train_on_pair(state, next_state, reward=1.0)
                policy_losses.append(result["policy_loss"])
                value_losses.append(result["value_loss"])

        # Also train on unsolved episodes with reward=0.0 (value head only has
        # meaningful signal, but we still run _train_on_pair for consistency).
        # However, unsolved episodes have empty trajectories, so nothing to do.

        avg_policy_loss = (
            sum(policy_losses) / len(policy_losses) if policy_losses else 0.0
        )
        avg_value_loss = (
            sum(value_losses) / len(value_losses) if value_losses else 0.0
        )

        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "solve_rate": solve_rate,
        }

    def _train_on_pair(self, state, target, reward: float) -> dict:
        """Train on a single (state, target) pair.

        Converts ``state`` and ``target`` to prefix tokens, builds
        the input sequence ``[BOS] state_tokens [SEP] target_tokens [EOS]``,
        computes policy loss (cross-entropy after SEP) and value loss
        (BCELoss on value head output vs reward).

        Parameters
        ----------
        state : sympy.Expr
            The current expression (e.g., an ``Integral``).
        target : sympy.Expr
            The next expression in the trajectory.
        reward : float
            The reward signal (1.0 for solved, 0.0 for unsolved).

        Returns
        -------
        dict
            ``{"policy_loss": float, "value_loss": float}``
        """
        # Step 1: Convert state and target to prefix tokens
        try:
            state_tokens = sympy_to_prefix(state)
            target_tokens = sympy_to_prefix(target)
        except (ValueError, TypeError):
            return {"policy_loss": 0.0, "value_loss": 0.0}

        # Step 2: Build [BOS] state_tokens [SEP] target_tokens [EOS]
        full_tokens = ["BOS"] + state_tokens + ["SEP"] + target_tokens + ["EOS"]

        # Step 3: Convert to token IDs
        token_ids = []
        for tok in full_tokens:
            tid = self.vocab.token_to_id(tok)
            if tid is None:
                # Unknown token; skip this pair
                return {"policy_loss": 0.0, "value_loss": 0.0}
            token_ids.append(tid)

        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        # Step 4: Build target IDs for policy loss
        # Target is the same sequence shifted by 1
        # We only compute loss on tokens after SEP (the target part)
        target_ids = token_ids[1:] + [self.vocab.pad_id]  # shifted by 1
        target_tensor = torch.tensor([target_ids], dtype=torch.long, device=self.device)

        # Mask: only compute loss after SEP position
        # Find SEP position
        sep_pos = len(["BOS"] + state_tokens)  # index of SEP in the sequence
        # Set all positions before and including SEP to -100 (ignore)
        for i in range(sep_pos + 1):
            target_tensor[0, i] = -100

        # Step 5: Forward pass
        sep_positions = torch.tensor([sep_pos], dtype=torch.long, device=self.device)
        logits, value = self.model(input_tensor, sep_positions)

        # Step 6: Policy loss (cross-entropy on shifted logits after SEP)
        shifted_logits = logits[:, :-1].contiguous()  # (1, T-1, vocab_size)
        shifted_targets = target_tensor[:, :-1].contiguous()  # (1, T-1)

        B, T_minus_1, V = shifted_logits.shape
        policy_loss = self.policy_loss_fn(
            shifted_logits.view(B * T_minus_1, V),
            shifted_targets.view(B * T_minus_1),
        )

        # Step 7: Value loss (BCELoss on value head output vs reward)
        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
        value_loss = self.value_loss_fn(value, reward_tensor)

        # Step 8: Backward pass and optimizer step
        total_loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }
