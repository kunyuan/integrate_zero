"""Tests for the supervised training loop."""

import torch
from integrate_zero.train.supervised import SupervisedTrainer
from integrate_zero.data.dataset import IntegrationDataset
from integrate_zero.model.transformer import IntegrateZeroModel
from integrate_zero.data.vocabulary import Vocabulary


def test_supervised_one_step():
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                               num_layers=2, d_ff=128)
    ds = IntegrationDataset(num_samples=20, max_depth=3)
    trainer = SupervisedTrainer(model, ds, vocab, batch_size=4, lr=1e-3)
    loss_before = trainer.evaluate_loss()["total_loss"]
    trainer.train_epoch()
    loss_after = trainer.evaluate_loss()["total_loss"]
    assert loss_after < loss_before


def test_supervised_saves_checkpoint(tmp_path):
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                               num_layers=2, d_ff=128)
    ds = IntegrationDataset(num_samples=20, max_depth=3)
    trainer = SupervisedTrainer(model, ds, vocab, batch_size=4, lr=1e-3)
    trainer.train_epoch()
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(str(path))
    assert path.exists()


def test_supervised_load_checkpoint(tmp_path):
    """Test that load_checkpoint restores model and optimizer state."""
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                               num_layers=2, d_ff=128)
    ds = IntegrationDataset(num_samples=20, max_depth=3)
    trainer = SupervisedTrainer(model, ds, vocab, batch_size=4, lr=1e-3)
    trainer.train_epoch()

    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(str(path))

    # Create a fresh model and trainer, then load
    model2 = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                                num_layers=2, d_ff=128)
    trainer2 = SupervisedTrainer(model2, ds, vocab, batch_size=4, lr=1e-3)
    trainer2.load_checkpoint(str(path))

    # Model weights should match after loading
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2), "Model parameters should match after loading checkpoint"


def test_evaluate_loss_returns_dict():
    """Test that evaluate_loss returns a dict with finite float values."""
    vocab = Vocabulary()
    model = IntegrateZeroModel(vocab_size=len(vocab), d_model=64, nhead=2,
                               num_layers=2, d_ff=128)
    ds = IntegrationDataset(num_samples=10, max_depth=3)
    trainer = SupervisedTrainer(model, ds, vocab, batch_size=4, lr=1e-3)
    result = trainer.evaluate_loss()
    assert isinstance(result, dict)
    for key in ("total_loss", "policy_loss", "value_loss"):
        assert key in result
        assert isinstance(result[key], float)
        assert result[key] > 0.0
        assert not torch.isnan(torch.tensor(result[key]))
        assert not torch.isinf(torch.tensor(result[key]))
