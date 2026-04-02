"""Evaluation module for IntegrateZero.

Provides the ``Evaluator`` class which measures model performance on a
list of integration problems using MCTS, and compares against a SymPy
baseline.

Usage::

    evaluator = Evaluator(model, vocab)
    results = evaluator.evaluate(problems, search_budget=200)
    baseline = evaluator.sympy_baseline(problems)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import sympy
from sympy import Integral

from integrate_zero.mcts.search import MCTS


class Evaluator:
    """Evaluate a model's integration capability on a set of problems.

    Parameters
    ----------
    model : object or None
        The neural network model used by MCTS for candidate generation
        and value estimation.  May be ``None`` when only using
        ``sympy_baseline()``.
    vocab : object or None
        The vocabulary for encoding/decoding expressions.  May be ``None``
        when only using ``sympy_baseline()``.
    """

    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab

    def evaluate(
        self,
        problems: list,
        search_budget: int = 200,
        num_candidates: int = 8,
        max_steps: int = 20,
    ) -> dict:
        """Evaluate the model on a list of integration problems using MCTS.

        For each problem, an MCTS search is run.  A problem is considered
        solved if ``mcts.search()`` returns a non-None trajectory (i.e.,
        the search found a terminal expression with no remaining Integral).

        Parameters
        ----------
        problems : list[sympy.Expr]
            A list of SymPy expressions to solve (typically ``Integral``
            objects).
        search_budget : int
            Total number of MCTS iterations per problem.
        num_candidates : int
            Number of candidate expressions to generate at each expansion.
        max_steps : int
            Maximum number of sequential transformation steps.

        Returns
        -------
        dict
            A dictionary with the following keys:

            - ``"solve_rate"`` (float): Fraction of problems solved.
            - ``"avg_steps"`` (float): Average number of steps for solved
              problems (0.0 if none solved).
            - ``"total"`` (int): Total number of problems.
            - ``"solved"`` (int): Number of problems solved.
        """
        total = len(problems)
        if total == 0:
            return {
                "solve_rate": 0.0,
                "avg_steps": 0.0,
                "total": 0,
                "solved": 0,
            }

        mcts = MCTS(
            model=self.model,
            vocab=self.vocab,
            max_steps=max_steps,
            num_candidates=num_candidates,
            search_budget=search_budget,
        )

        solved = 0
        total_steps = 0

        # Run MCTS searches in parallel using threads.
        # GPU inference is thread-safe; SymPy verification uses ProcessPool.
        with ThreadPoolExecutor(max_workers=min(len(problems), 16)) as executor:
            trajectories = list(executor.map(mcts.search, problems))

        for trajectory in trajectories:
            if trajectory is not None:
                solved += 1
                # trajectory includes the initial problem, so steps = len - 1
                total_steps += len(trajectory) - 1

        solve_rate = solved / total
        avg_steps = total_steps / solved if solved > 0 else 0.0

        return {
            "solve_rate": solve_rate,
            "avg_steps": avg_steps,
            "total": total,
            "solved": solved,
        }

    def sympy_baseline(self, problems: list) -> dict:
        """Evaluate SymPy's built-in ``integrate()`` as a baseline.

        For each problem (an ``Integral`` expression), calls ``problem.doit()``
        and checks whether the result contains any remaining ``Integral``
        sub-expressions.  If not, the problem is counted as solved.

        Parameters
        ----------
        problems : list[sympy.Expr]
            A list of SymPy ``Integral`` expressions.

        Returns
        -------
        dict
            A dictionary with the following keys:

            - ``"solve_rate"`` (float): Fraction of problems solved.
            - ``"total"`` (int): Total number of problems.
            - ``"solved"`` (int): Number of problems solved.
        """
        total = len(problems)
        if total == 0:
            return {
                "solve_rate": 0.0,
                "total": 0,
                "solved": 0,
            }

        solved = 0
        for problem in problems:
            try:
                result = problem.doit()
                if not result.has(Integral):
                    solved += 1
            except Exception:
                # If SymPy raises during integration, count as unsolved
                pass

        solve_rate = solved / total

        return {
            "solve_rate": solve_rate,
            "total": total,
            "solved": solved,
        }
