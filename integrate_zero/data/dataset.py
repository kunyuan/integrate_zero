"""PyTorch Dataset for symbolic integration training pairs.

Generates (f, F) pairs offline, tokenizes them to prefix notation,
and produces input/target sequences for supervised training.

Data format:
    Input:   [BOS] f_tokens [SEP] F_tokens [EOS]
    Target:  [-100 ... -100] F_tokens [EOS]
             (no loss on positions up to and including SEP)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from integrate_zero.data.vocabulary import Vocabulary
from integrate_zero.data.generator import generate_training_pair
from integrate_zero.data.prefix import sympy_to_prefix

logger = logging.getLogger(__name__)


class IntegrationDataset(Dataset):
    """Dataset of symbolic integration training pairs.

    Each sample is a pair ``(f, F)`` where ``f = dF/dx``, tokenized into
    prefix notation and encoded as integer token IDs.

    Parameters
    ----------
    num_samples : int
        Number of training pairs to generate.
    max_depth : int
        Maximum depth of randomly generated expression trees.
    max_len : int, optional
        Maximum sequence length (BOS + f_tokens + SEP + F_tokens + EOS).
        Samples exceeding this length are discarded. Defaults to 512.
    """

    def __init__(
        self,
        num_samples: int,
        max_depth: int = 3,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.vocab = Vocabulary()
        self.max_len = max_len
        self.samples: List[Dict[str, torch.Tensor]] = []

        self._generate_samples(num_samples, max_depth)

    def _generate_samples(self, num_samples: int, max_depth: int) -> None:
        """Generate and tokenize training pairs, storing valid ones."""
        failures = 0
        attempts = 0
        max_attempts = num_samples * 5  # allow plenty of retries

        while len(self.samples) < num_samples and attempts < max_attempts:
            attempts += 1
            try:
                f, F = generate_training_pair(max_depth=max_depth)

                # Convert to prefix tokens
                f_tokens = sympy_to_prefix(f)
                F_tokens = sympy_to_prefix(F)

                # Filter out samples with tokens not in vocabulary
                if not self._all_tokens_in_vocab(f_tokens):
                    failures += 1
                    continue
                if not self._all_tokens_in_vocab(F_tokens):
                    failures += 1
                    continue

                # Build the full sequence: [BOS] f_tokens [SEP] F_tokens [EOS]
                sample = self._build_sample(f_tokens, F_tokens)
                if sample is not None:
                    self.samples.append(sample)

            except Exception:
                failures += 1
                continue

        if len(self.samples) < num_samples:
            logger.warning(
                "IntegrationDataset: requested %d samples but only generated %d "
                "(%d failures in %d attempts)",
                num_samples,
                len(self.samples),
                failures,
                attempts,
            )

    def _all_tokens_in_vocab(self, tokens: List[str]) -> bool:
        """Check that every token is in the vocabulary."""
        for token in tokens:
            if token not in self.vocab:
                return False
        return True

    def _build_sample(
        self, f_tokens: List[str], F_tokens: List[str]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Build input_ids, target_ids, and value_label from token lists.

        Returns None if the sequence exceeds max_len.
        """
        bos_id = self.vocab.bos_id
        sep_id = self.vocab.sep_id
        eos_id = self.vocab.eos_id

        # Convert tokens to IDs
        f_ids = [self.vocab.token_to_id(t) for t in f_tokens]
        F_ids = [self.vocab.token_to_id(t) for t in F_tokens]

        # Full sequence: [BOS] f_ids [SEP] F_ids [EOS]
        input_ids = [bos_id] + f_ids + [sep_id] + F_ids + [eos_id]

        # Check max length
        if len(input_ids) > self.max_len:
            return None

        # Target IDs: -100 for positions up to and including SEP,
        # actual IDs for F_tokens and EOS
        num_masked = 1 + len(f_ids) + 1  # BOS + f_tokens + SEP
        target_ids = [-100] * num_masked + F_ids + [eos_id]

        assert len(input_ids) == len(target_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "value_label": 1.0,
        }

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return a single sample by index.

        Returns
        -------
        dict
            ``input_ids``:   LongTensor of shape ``(seq_len,)``
            ``target_ids``:  LongTensor of shape ``(seq_len,)``
            ``value_label``: float (1.0 for all generated pairs)
        """
        return self.samples[idx]

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Pad a batch of samples to the same length.

        Parameters
        ----------
        batch : list[dict]
            List of samples from ``__getitem__``.

        Returns
        -------
        dict
            ``input_ids``:   LongTensor ``(batch_size, max_seq_len)``
                             padded with PAD (id=0)
            ``target_ids``:  LongTensor ``(batch_size, max_seq_len)``
                             padded with -100
            ``value_label``: FloatTensor ``(batch_size,)``
        """
        max_seq_len = max(item["input_ids"].size(0) for item in batch)

        input_ids_padded = []
        target_ids_padded = []
        value_labels = []

        pad_id = 0  # PAD token ID

        for item in batch:
            seq_len = item["input_ids"].size(0)
            pad_len = max_seq_len - seq_len

            # Pad input_ids with PAD (0)
            input_padded = torch.cat([
                item["input_ids"],
                torch.full((pad_len,), pad_id, dtype=torch.long),
            ])
            input_ids_padded.append(input_padded)

            # Pad target_ids with -100
            target_padded = torch.cat([
                item["target_ids"],
                torch.full((pad_len,), -100, dtype=torch.long),
            ])
            target_ids_padded.append(target_padded)

            value_labels.append(item["value_label"])

        return {
            "input_ids": torch.stack(input_ids_padded),
            "target_ids": torch.stack(target_ids_padded),
            "value_label": torch.tensor(value_labels, dtype=torch.float),
        }
