"""Supervised training loop for the IntegrateZero model.

Trains the model on (integrand, antiderivative) pairs using:
- Policy loss: CrossEntropy on next-token prediction (shifted by 1)
- Value loss: BCE on the value head output vs ground-truth labels
- Total loss = policy_loss + value_loss
"""

from __future__ import annotations

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
    val_dataset : IntegrationDataset or None
        Optional separate validation dataset.  If ``None``, evaluation
        is performed on the training dataset.
    """

    def __init__(
        self,
        model: IntegrateZeroModel,
        dataset: IntegrationDataset,
        vocab: Vocabulary,
        batch_size: int = 256,
        lr: float = 1e-4,
        val_dataset: IntegrationDataset | None = None,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.vocab = vocab
        self.batch_size = batch_size

        self.device = next(model.parameters()).device

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.policy_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.value_loss_fn = nn.BCELoss()

        self.sep_id = vocab.sep_id

    def _make_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Create a DataLoader over the dataset."""
        num_workers = 4
        use_pin_memory = self.device.type == "cuda"
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=IntegrationDataset.collate_fn,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=num_workers > 0,
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

    def train_epoch(self) -> dict[str, float]:
        """Train for one full epoch over the dataset.

        Returns
        -------
        dict
            Keys: ``total_loss``, ``policy_loss``, ``value_loss`` —
            average values over the epoch.
        """
        self.model.train()
        dataloader = self._make_dataloader(shuffle=True)

        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        num_batches = 0

        for batch in dataloader:
            self.optimizer.zero_grad()
            total_loss, policy_loss, value_loss = self._compute_loss(batch)
            total_loss.backward()
            self.optimizer.step()

            total_loss_sum += total_loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            num_batches += 1

        n = max(num_batches, 1)
        return {
            "total_loss": total_loss_sum / n,
            "policy_loss": policy_loss_sum / n,
            "value_loss": value_loss_sum / n,
        }

    @torch.no_grad()
    def evaluate_loss(self) -> dict[str, float]:
        """Evaluate the average loss without training.

        Uses the validation dataset if one was provided, otherwise
        evaluates on the training dataset.

        Returns
        -------
        dict
            Keys: ``total_loss``, ``policy_loss``, ``value_loss`` —
            average values over the evaluation set.
        """
        self.model.eval()
        eval_ds = self.val_dataset if self.val_dataset is not None else self.dataset
        num_workers = 4
        use_pin_memory = self.device.type == "cuda"
        dataloader = DataLoader(
            eval_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=IntegrationDataset.collate_fn,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=num_workers > 0,
        )

        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        num_batches = 0

        for batch in dataloader:
            total_loss, policy_loss, value_loss = self._compute_loss(batch)
            total_loss_sum += total_loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            num_batches += 1

        n = max(num_batches, 1)
        return {
            "total_loss": total_loss_sum / n,
            "policy_loss": policy_loss_sum / n,
            "value_loss": value_loss_sum / n,
        }

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
