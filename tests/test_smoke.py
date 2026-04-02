"""End-to-end smoke test for the IntegrateZero pipeline.

Runs the full pipeline with tiny parameters to verify everything
connects without crashes:
1. Generate training pairs
2. Supervised training (2 epochs)
3. Verify loss decreases
4. MCTS search on simple problems
5. RL training (1 iteration)
6. Evaluation + SymPy baseline
"""

import random

import pytest
import sympy
import torch
from sympy import Integral, Symbol

from integrate_zero.data.dataset import IntegrationDataset
from integrate_zero.data.vocabulary import Vocabulary
from integrate_zero.eval.evaluate import Evaluator
from integrate_zero.eval.textbook import get_textbook_problems
from integrate_zero.mcts.search import MCTS
from integrate_zero.model.transformer import IntegrateZeroModel
from integrate_zero.train.rl import RLTrainer
from integrate_zero.train.supervised import SupervisedTrainer

x = Symbol("x")


def _make_tiny_model(vocab: Vocabulary) -> IntegrateZeroModel:
    """Create a tiny model for smoke testing."""
    return IntegrateZeroModel(
        vocab_size=len(vocab),
        d_model=64,
        nhead=2,
        num_layers=2,
        d_ff=128,
        max_seq_len=256,
    )


def _simple_problems(n: int = 5) -> list:
    """Return simple integration problems for smoke testing."""
    problems = [
        Integral(x**2, x),
        Integral(x, x),
        Integral(sympy.sin(x), x),
        Integral(sympy.cos(x), x),
        Integral(sympy.exp(x), x),
        Integral(x**3, x),
        Integral(2*x + 1, x),
    ]
    return problems[:n]


@pytest.fixture
def tiny_setup():
    """Set up a tiny model and dataset for smoke testing."""
    random.seed(42)
    torch.manual_seed(42)

    vocab = Vocabulary()
    model = _make_tiny_model(vocab)
    dataset = IntegrationDataset(num_samples=50, max_depth=3, seed=42)

    return model, vocab, dataset


class TestSmokeEndToEnd:
    """End-to-end smoke tests with tiny parameters."""

    def test_data_generation(self):
        """Step 1: Generate training pairs with a seed."""
        ds = IntegrationDataset(num_samples=50, max_depth=3, seed=42)
        assert len(ds) > 0
        sample = ds[0]
        assert "input_ids" in sample
        assert "target_ids" in sample
        assert "value_label" in sample
        assert isinstance(sample["input_ids"], torch.Tensor)
        assert sample["input_ids"].dtype == torch.long

    def test_data_reproducibility(self):
        """Verify that seeded generation produces identical datasets."""
        ds1 = IntegrationDataset(num_samples=20, max_depth=3, seed=123)
        ds2 = IntegrationDataset(num_samples=20, max_depth=3, seed=123)
        assert len(ds1) == len(ds2)
        for i in range(len(ds1)):
            assert torch.equal(ds1[i]["input_ids"], ds2[i]["input_ids"])

    def test_data_save_load(self, tmp_path):
        """Verify dataset save/load roundtrip."""
        ds = IntegrationDataset(num_samples=20, max_depth=3, seed=42)
        path = tmp_path / "test_ds.pt"
        ds.save(path)

        ds_loaded = IntegrationDataset.load(path)
        assert len(ds_loaded) == len(ds)
        for i in range(len(ds)):
            assert torch.equal(ds[i]["input_ids"], ds_loaded[i]["input_ids"])
            assert torch.equal(ds[i]["target_ids"], ds_loaded[i]["target_ids"])

    def test_supervised_training(self, tiny_setup):
        """Steps 2-3: Supervised training for 2 epochs, verify loss decreases."""
        model, vocab, dataset = tiny_setup
        trainer = SupervisedTrainer(
            model, dataset, vocab,
            batch_size=4, lr=1e-3,
        )

        metrics_before = trainer.evaluate_loss()
        trainer.train_epoch()
        trainer.train_epoch()
        metrics_after = trainer.evaluate_loss()

        loss_before = metrics_before["total_loss"]
        loss_after = metrics_after["total_loss"]
        assert isinstance(loss_before, float)
        assert isinstance(loss_after, float)
        assert loss_before > 0.0
        assert loss_after > 0.0
        assert loss_after < loss_before, (
            f"Loss should decrease: {loss_before:.4f} -> {loss_after:.4f}"
        )

    def test_mcts_search(self, tiny_setup):
        """Step 4: MCTS search on simple problems (no crashes expected)."""
        model, vocab, _dataset = tiny_setup
        model.eval()

        mcts = MCTS(
            model=model,
            vocab=vocab,
            max_steps=3,
            num_candidates=4,
            search_budget=10,
        )

        problems = _simple_problems(5)
        for problem in problems:
            # We don't expect an untrained model to solve anything,
            # but it should not crash.
            trajectory = mcts.search(problem)
            assert trajectory is None or isinstance(trajectory, list)

    def test_rl_training(self, tiny_setup):
        """Step 5: RL training with 1 iteration on simple problems."""
        model, vocab, _dataset = tiny_setup

        rl_trainer = RLTrainer(
            model, vocab,
            num_candidates=4,
            search_budget=10,
            max_steps=3,
            lr=1e-3,
        )

        problems = _simple_problems(3)
        stats = rl_trainer.train_iteration(problems)

        assert isinstance(stats, dict)
        assert "policy_loss" in stats
        assert "value_loss" in stats
        assert "solve_rate" in stats
        assert isinstance(stats["solve_rate"], float)
        assert 0.0 <= stats["solve_rate"] <= 1.0

    def test_evaluation(self, tiny_setup):
        """Step 6: Evaluation on problems + SymPy baseline."""
        model, vocab, _dataset = tiny_setup

        evaluator = Evaluator(model, vocab)

        problems = _simple_problems(5)

        # Model evaluation (untrained, expect 0 solves but no crashes)
        model_results = evaluator.evaluate(
            problems,
            search_budget=10,
            num_candidates=4,
            max_steps=3,
        )
        assert isinstance(model_results, dict)
        assert "solve_rate" in model_results
        assert "avg_steps" in model_results
        assert "total" in model_results
        assert "solved" in model_results
        assert model_results["total"] == 5

        # SymPy baseline
        sympy_results = evaluator.sympy_baseline(problems)
        assert isinstance(sympy_results, dict)
        assert "solve_rate" in sympy_results
        assert sympy_results["total"] == 5
        # SymPy should solve these simple integrals
        assert sympy_results["solved"] > 0

    def test_textbook_problems_load(self):
        """Verify textbook problems can be loaded."""
        problems = get_textbook_problems()
        assert len(problems) >= 20
        for p in problems:
            assert isinstance(p, Integral)

    def test_full_pipeline(self, tiny_setup, tmp_path):
        """Full pipeline: generate -> train -> save -> load -> eval."""
        model, vocab, dataset = tiny_setup

        # Supervised train
        trainer = SupervisedTrainer(
            model, dataset, vocab,
            batch_size=4, lr=1e-3,
        )
        loss1 = trainer.evaluate_loss()["total_loss"]
        trainer.train_epoch()
        loss2 = trainer.evaluate_loss()["total_loss"]
        assert loss2 < loss1

        # Save/load checkpoint
        ckpt_path = tmp_path / "smoke_ckpt.pt"
        trainer.save_checkpoint(str(ckpt_path))
        assert ckpt_path.exists()

        model2 = _make_tiny_model(vocab)
        trainer2 = SupervisedTrainer(
            model2, dataset, vocab,
            batch_size=4, lr=1e-3,
        )
        trainer2.load_checkpoint(str(ckpt_path))

        # Loaded model should have same eval loss
        loss_loaded = trainer2.evaluate_loss()["total_loss"]
        assert abs(loss_loaded - loss2) < 1e-4

        # RL iteration
        rl_trainer = RLTrainer(
            model2, vocab,
            num_candidates=4,
            search_budget=10,
            max_steps=3,
            lr=1e-3,
        )
        problems = _simple_problems(2)
        stats = rl_trainer.train_iteration(problems)
        assert isinstance(stats["solve_rate"], float)

        # Eval
        evaluator = Evaluator(model2, vocab)
        results = evaluator.evaluate(
            problems,
            search_budget=10,
            num_candidates=4,
            max_steps=3,
        )
        assert results["total"] == 2
