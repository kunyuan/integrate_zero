"""Supervised training loop for the IntegrateZero model.

Trains the model on (integrand, antiderivative) pairs using:
- Policy loss: CrossEntropy on next-token prediction (shifted by 1)
- Value loss: BCE on the value head output vs ground-truth labels
- Total loss = policy_loss + value_loss
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from integrate_zero.data.dataset import IntegrationDataset
from integrate_zero.data.vocabulary import Vocabulary
from integrate_zero.model.transformer import IntegrateZeroModel


class SupervisedTrainer:
    """Supervised trainer for the IntegrateZero transformer model.

    Parameters
    ----------
    model : IntegrateZeroModel
        The transformer model with policy (logits) and value heads.
    dataset : IntegrationDataset
        Training dataset providing (input_ids, target_ids, value_label) samples.
    vocab : Vocabulary
        Vocabulary instance (used to look up the SEP token ID).
    batch_size : int
        Mini-batch size for training and evaluation.
    lr : float
        Learning rate for the Adam optimizer.
    """

    def __init__(
        self,
        model: IntegrateZeroModel,
        dataset: IntegrationDataset,
        vocab: Vocabulary,
        batch_size: int = 256,
        lr: float = 1e-4,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.vocab = vocab
        self.batch_size = batch_size

        self.device = next(model.parameters()).device

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.policy_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.value_loss_fn = nn.BCELoss()

        self.sep_id = vocab.sep_id

    def _make_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Create a DataLoader over the dataset."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=IntegrationDataset.collate_fn,
        )

    def _find_sep_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Find the position of the first SEP token in each sequence.

        Parameters
        ----------
        input_ids : Tensor, shape ``(B, T)``

        Returns
        -------
        Tensor, shape ``(B,)``
            Index of the first SEP token in each sequence.
        """
        return (input_ids == self.sep_id).long().argmax(dim=1)

    def _compute_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute policy and value losses for a batch.

        Returns
        -------
        total_loss, policy_loss, value_loss : Tensor
            Scalar loss tensors.
        """
        input_ids = batch["input_ids"].to(self.device)
        target_ids = batch["target_ids"].to(self.device)
        value_labels = batch["value_label"].to(self.device)

        sep_positions = self._find_sep_positions(input_ids)

        logits, value = self.model(input_ids, sep_positions)

        # Policy loss: next-token prediction (shift by 1)
        # logits[:, :-1] predicts the token at position t+1
        # target_ids[:, 1:] is the ground truth for position t+1
        shifted_logits = logits[:, :-1].contiguous()  # (B, T-1, vocab_size)
        shifted_targets = target_ids[:, 1:].contiguous()  # (B, T-1)

        # Reshape for CrossEntropyLoss: (B*(T-1), vocab_size) vs (B*(T-1),)
        B, T_minus_1, V = shifted_logits.shape
        policy_loss = self.policy_loss_fn(
            shifted_logits.view(B * T_minus_1, V),
            shifted_targets.view(B * T_minus_1),
        )

        # Value loss: BCE on value head output vs value_labels
        value_loss = self.value_loss_fn(value, value_labels)

        total_loss = policy_loss + value_loss

        return total_loss, policy_loss, value_loss

    def train_epoch(self) -> float:
        """Train for one full epoch over the dataset.

        Returns
        -------
        float
            Average total loss over the epoch.
        """
        self.model.train()
        dataloader = self._make_dataloader(shuffle=True)

        total_loss_sum = 0.0
        num_batches = 0

        for batch in dataloader:
            self.optimizer.zero_grad()
            total_loss, _, _ = self._compute_loss(batch)
            total_loss.backward()
            self.optimizer.step()

            total_loss_sum += total_loss.item()
            num_batches += 1

        return total_loss_sum / max(num_batches, 1)

    @torch.no_grad()
    def evaluate_loss(self) -> float:
        """Evaluate the average loss over the entire dataset without training.

        Returns
        -------
        float
            Average total loss.
        """
        self.model.eval()
        dataloader = self._make_dataloader(shuffle=False)

        total_loss_sum = 0.0
        num_batches = 0

        for batch in dataloader:
            total_loss, _, _ = self._compute_loss(batch)
            total_loss_sum += total_loss.item()
            num_batches += 1

        return total_loss_sum / max(num_batches, 1)

    def save_checkpoint(self, path: str) -> None:
        """Save model and optimizer state to a file.

        Parameters
        ----------
        path : str
            File path for the checkpoint.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model and optimizer state from a file.

        Parameters
        ----------
        path : str
            File path of the checkpoint to load.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
