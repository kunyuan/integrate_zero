"""Monte Carlo Tree Search for symbolic integration.

This module implements the MCTS algorithm used by IntegrateZero to search
for multi-step integration solutions.  Each node in the search tree
represents a SymPy expression (possibly containing unevaluated ``Integral``
sub-expressions).  Actions correspond to verified transformation steps
(integration or identity rewrites).

The ``MCTS`` class orchestrates the search loop:

1. Select a leaf via UCB.
2. Expand the leaf by generating candidate expressions from the model.
3. Evaluate the leaf with the value head.
4. Backpropagate the value up to the root.

After a configurable search budget is spent, the most-visited child is
chosen as the next step, and the search continues from there until the
expression is terminal (no remaining ``Integral``) or the step limit is
reached.

Model-dependent methods (``_generate_candidates`` and ``_evaluate``) are
placeholders in this module and will be connected to the neural network in
Task 11.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import sympy
import torch
from sympy import Symbol

from integrate_zero.data.prefix import prefix_to_sympy, sympy_to_prefix
from integrate_zero.verify.verifier import StepType, is_terminal, verify_step


class MCTSNode:
    """A single node in the MCTS search tree.

    Parameters
    ----------
    state : sympy.Expr
        The SymPy expression at this node.
    prior : float
        The prior probability P(action) from the policy network.
    parent : MCTSNode or None
        The parent node (None for the root).
    step_type : StepType or None
        The type of transformation step that led to this node.
    """

    def __init__(
        self,
        state: sympy.Expr,
        prior: float = 1.0,
        parent: Optional["MCTSNode"] = None,
        step_type: Optional[StepType] = None,
    ):
        self.state = state
        self.prior = prior
        self.parent = parent
        self.step_type = step_type
        self.children: List["MCTSNode"] = []
        self.visit_count: int = 0
        self.value_sum: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def q_value(self) -> float:
        """Mean value estimate (value_sum / visit_count), 0 if unvisited."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    # ------------------------------------------------------------------
    # UCB
    # ------------------------------------------------------------------

    def ucb_score(self, c_puct: float = 1.5) -> float:
        """Compute the Upper Confidence Bound score.

        Formula::

            score = q_value + c_puct * prior * sqrt(N_parent) / (1 + N_child)

        Parameters
        ----------
        c_puct : float
            Exploration constant.

        Returns
        -------
        float
            The UCB score for this node.
        """
        n_parent = self.parent.visit_count if self.parent is not None else 0
        exploration = c_puct * self.prior * math.sqrt(n_parent) / (1 + self.visit_count)
        return self.q_value + exploration

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def expand_with(
        self,
        candidate_expr: sympy.Expr,
        prior: float,
        x: Symbol = Symbol("x"),
    ) -> Optional["MCTSNode"]:
        """Attempt to expand this node with a candidate expression.

        The candidate is verified against the current state using
        ``verify_step``.  If the step is valid (INTEGRATION or IDENTITY),
        a new child node is created and appended.  If the step is INVALID,
        ``None`` is returned and no child is added.

        Parameters
        ----------
        candidate_expr : sympy.Expr
            The proposed next expression.
        prior : float
            The prior probability for this action.
        x : Symbol
            The integration variable.

        Returns
        -------
        MCTSNode or None
            The newly created child node, or None if the step is invalid.
        """
        step_type = verify_step(self.state, candidate_expr, x)
        if step_type == StepType.INVALID:
            return None
        child = MCTSNode(
            state=candidate_expr,
            prior=prior,
            parent=self,
            step_type=step_type,
        )
        self.children.append(child)
        return child

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def backpropagate(self, value: float) -> None:
        """Backpropagate a value from this node up to the root.

        Increments ``visit_count`` and adds ``value`` to ``value_sum``
        at each ancestor node (including this one).

        Parameters
        ----------
        value : float
            The evaluation value to propagate.
        """
        node: Optional["MCTSNode"] = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def best_child(self, c_puct: float = 1.5) -> Optional["MCTSNode"]:
        """Return the child with the highest UCB score.

        Parameters
        ----------
        c_puct : float
            Exploration constant passed to :meth:`ucb_score`.

        Returns
        -------
        MCTSNode or None
            The best child, or None if there are no children.
        """
        if not self.children:
            return None
        return max(self.children, key=lambda c: c.ucb_score(c_puct))

    def is_terminal(self) -> bool:
        """Check whether this node's state contains no ``Integral``.

        Returns
        -------
        bool
            True if the expression has no remaining integrals.
        """
        return is_terminal(self.state)


class MCTS:
    """Monte Carlo Tree Search for multi-step symbolic integration.

    Parameters
    ----------
    model : object
        The neural network model (used by ``_generate_candidates`` and
        ``_evaluate``).  May be ``None`` when using placeholder methods.
    vocab : object or None
        Vocabulary for encoding/decoding expressions.
    max_steps : int
        Maximum number of sequential transformation steps.
    num_candidates : int
        Number of candidate expressions to generate at each expansion.
    search_budget : int
        Total number of MCTS iterations across all steps.
    c_puct : float
        Exploration constant for the UCB formula.
    """

    def __init__(
        self,
        model,
        vocab=None,
        max_steps: int = 20,
        num_candidates: int = 8,
        search_budget: int = 200,
        c_puct: float = 1.5,
    ):
        self.model = model
        self.vocab = vocab
        self.max_steps = max_steps
        self.num_candidates = num_candidates
        self.search_budget = search_budget
        self.c_puct = c_puct

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, problem: sympy.Expr) -> Optional[List[sympy.Expr]]:
        """Run multi-step MCTS from the given problem expression.

        At each step, a fixed number of MCTS iterations
        (``search_budget // max_steps``) are performed to decide the
        next transformation.  If the expression becomes terminal (no
        remaining ``Integral``), the solution trajectory is returned.

        Parameters
        ----------
        problem : sympy.Expr
            The starting expression (typically an ``Integral``).

        Returns
        -------
        list[sympy.Expr] or None
            The sequence of expressions from *problem* to the terminal
            solution, or ``None`` if no solution was found within the
            budget.
        """
        root = MCTSNode(state=problem, prior=1.0)
        trajectory = [problem]
        iterations_per_step = max(1, self.search_budget // self.max_steps)

        for _ in range(self.max_steps):
            # Run search iterations at the current node
            best = self._run_search(root, iterations_per_step)
            if best is None:
                return None

            trajectory.append(best.state)

            if best.is_terminal():
                return trajectory

            # Advance the root to the best child
            root = best

        return None

    # ------------------------------------------------------------------
    # Internal search loop
    # ------------------------------------------------------------------

    def _run_search(
        self, root: MCTSNode, iterations: int
    ) -> Optional[MCTSNode]:
        """Run a fixed number of MCTS iterations from *root*.

        Each iteration follows the select-expand-evaluate-backpropagate
        cycle.  After all iterations, returns the child of *root* with
        the highest visit count (the most robust choice).

        Parameters
        ----------
        root : MCTSNode
            The current root of the sub-tree.
        iterations : int
            Number of MCTS iterations to perform.

        Returns
        -------
        MCTSNode or None
            The most-visited child of *root*, or None if *root* has no
            children after all iterations.
        """
        for _ in range(iterations):
            leaf = self._select(root)
            self._expand(leaf)
            value = self._evaluate(leaf)
            leaf.backpropagate(value)

        if not root.children:
            return None
        # Select the most-visited child (robust selection)
        return max(root.children, key=lambda c: c.visit_count)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Navigate the tree following UCB to find a leaf node.

        A leaf is a node with no children.

        Parameters
        ----------
        node : MCTSNode
            The starting node (usually the current root).

        Returns
        -------
        MCTSNode
            The selected leaf node.
        """
        while node.children:
            node = node.best_child(self.c_puct)
        return node

    def _expand(self, node: MCTSNode) -> None:
        """Expand a leaf node by generating and verifying candidates.

        Calls ``_generate_candidates`` to get proposed expressions and
        their priors, then uses ``MCTSNode.expand_with`` to verify each
        candidate and add valid ones as children.

        Parameters
        ----------
        node : MCTSNode
            The leaf node to expand.
        """
        candidates = self._generate_candidates(node.state)
        for expr, prior in candidates:
            node.expand_with(expr, prior)

    # ------------------------------------------------------------------
    # Placeholders for model integration (Task 11)
    # ------------------------------------------------------------------

    def _generate_candidates(
        self, state: sympy.Expr
    ) -> List[Tuple[sympy.Expr, float]]:
        """Generate candidate next-expressions from the model.

        Bridges the neural network (token generation) with MCTS (SymPy
        expressions):

        1. Convert ``state`` to prefix tokens via ``sympy_to_prefix()``.
        2. Build prompt: ``[BOS] state_tokens [SEP]``.
        3. Convert to tensor IDs using the vocabulary.
        4. Run ``model.generate()`` multiple times with temperature sampling.
        5. For each generated sequence, extract tokens after [SEP] and
           before [EOS], then convert back to SymPy via ``prefix_to_sympy()``.
        6. Return list of ``(sympy_expr, prior)`` tuples. Uses a uniform
           prior of ``1.0 / num_candidates``.

        If a generated sequence cannot be parsed back to SymPy it is
        silently skipped.

        Parameters
        ----------
        state : sympy.Expr
            The current expression.

        Returns
        -------
        list[tuple[sympy.Expr, float]]
            Pairs of (candidate_expression, prior_probability).
        """
        # Fallback: if no model/vocab is available, return empty list
        if self.model is None or self.vocab is None:
            return []

        # Step 1: Convert state to prefix tokens
        try:
            state_tokens = sympy_to_prefix(state)
        except (ValueError, TypeError):
            return []

        # Step 2: Build prompt [BOS] state_tokens [SEP]
        prompt_tokens = ["BOS"] + state_tokens + ["SEP"]

        # Step 3: Convert to tensor IDs
        prompt_ids = []
        for tok in prompt_tokens:
            tid = self.vocab.token_to_id(tok)
            if tid is None:
                return []  # Unknown token in the state -- cannot proceed
            prompt_ids.append(tid)

        device = next(self.model.parameters()).device
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # Step 4 & 5: Generate multiple candidates
        uniform_prior = 1.0 / self.num_candidates
        candidates: List[Tuple[sympy.Expr, float]] = []

        sep_id = self.vocab.sep_id
        eos_id = self.vocab.eos_id

        for _ in range(self.num_candidates):
            try:
                generated = self.model.generate(
                    prompt=prompt_tensor,
                    max_new_tokens=50,
                    vocab=self.vocab,
                    temperature=1.0,
                    eos_id=eos_id,
                )
            except Exception:
                continue

            # Extract the full sequence of token IDs
            seq = generated[0].tolist()

            # Find the last SEP position and extract tokens after it
            last_sep_idx = -1
            for i, tid in enumerate(seq):
                if tid == sep_id:
                    last_sep_idx = i

            if last_sep_idx == -1:
                continue

            # Tokens after SEP and before EOS
            after_sep = seq[last_sep_idx + 1:]
            output_tokens = []
            for tid in after_sep:
                if tid == eos_id:
                    break
                tok = self.vocab.id_to_token(tid)
                if tok is None:
                    break
                output_tokens.append(tok)

            if not output_tokens:
                continue

            # Step 6: Convert back to SymPy
            try:
                expr = prefix_to_sympy(output_tokens)
            except (ValueError, TypeError, IndexError):
                continue

            candidates.append((expr, uniform_prior))

        return candidates

    def _evaluate(self, node: MCTSNode) -> float:
        """Evaluate a node using the value head of the model.

        For terminal nodes (no remaining ``Integral``), returns 1.0
        immediately.  Otherwise, converts the node's state to prefix
        tokens, builds an input tensor ``[BOS] state_tokens [SEP]``,
        runs the model forward pass, and returns the value head output
        (sigmoid probability at the SEP position) as a float.

        Falls back to 0.5 if no model or vocabulary is available.

        Parameters
        ----------
        node : MCTSNode
            The node to evaluate.

        Returns
        -------
        float
            The estimated value of the node's state (in [0, 1]).
        """
        # Fallback: if no model/vocab is available, return placeholder
        if self.model is None or self.vocab is None:
            return 0.5

        # Terminal nodes are solved -- value is 1.0
        if node.is_terminal():
            return 1.0

        # Step 1: Convert state to prefix tokens
        try:
            state_tokens = sympy_to_prefix(node.state)
        except (ValueError, TypeError):
            return 0.5

        # Step 2: Build input [BOS] state_tokens [SEP]
        input_tokens = ["BOS"] + state_tokens + ["SEP"]

        # Step 3: Convert to tensor IDs
        input_ids = []
        for tok in input_tokens:
            tid = self.vocab.token_to_id(tok)
            if tid is None:
                return 0.5  # Unknown token -- cannot evaluate
            input_ids.append(tid)

        device = next(self.model.parameters()).device
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        # Step 4: Run model forward pass (no gradient needed for search)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            _logits, value = self.model(input_tensor)

        if was_training:
            self.model.train()

        # Step 5: Return the value head output as a float
        return value.item()
