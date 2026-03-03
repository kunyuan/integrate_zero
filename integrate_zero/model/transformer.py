"""Decoder-only (GPT-style) Transformer with a value head.

The model encodes a prefix-notation mathematical expression as:

    [BOS] A_tokens [SEP] B_tokens [EOS]

and provides two outputs:

1. **Logits** for autoregressive token prediction (shape ``(B, T, vocab_size)``).
2. **Value** from an MLP head applied to the hidden state at the [SEP]
   position (shape ``(B,)``, in [0, 1]).

Implementation uses ``nn.TransformerDecoder`` as the backbone.  Since this is
decoder-only (no separate encoder), a dummy all-zeros memory tensor is passed
to the cross-attention layers.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from integrate_zero.model.decoding import ArityMask


class IntegrateZeroModel(nn.Module):
    """Decoder-only Transformer with a value head for symbolic integration.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary.
    d_model : int
        Dimensionality of token / positional embeddings and hidden states.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of ``TransformerDecoderLayer`` blocks.
    d_ff : int
        Dimensionality of the feed-forward inner layer.
    max_seq_len : int
        Maximum sequence length (for positional embeddings).
    dropout : float
        Dropout probability used throughout.
    pad_id : int
        Token ID used for padding (masked in attention).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        nhead: int = 6,
        num_layers: int = 8,
        d_ff: int = 1536,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        # ---- Embeddings ----
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.embed_dropout = nn.Dropout(dropout)

        # ---- Transformer Decoder ----
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # ---- Output heads ----
        # LayerNorm before final projection (as specified)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Value head: hidden state at [SEP] -> MLP (d_model -> 256 -> 1, sigmoid)
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialization
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform for linear layers and
        normal distribution for embeddings."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        # Zero out padding embedding
        with torch.no_grad():
            self.token_embedding.weight[self.pad_id].zero_()
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2 and "embedding" not in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    # ------------------------------------------------------------------
    # Mask construction
    # ------------------------------------------------------------------

    @staticmethod
    def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Create an upper-triangular causal mask.

        Returns a ``(seq_len, seq_len)`` float tensor where masked positions
        are ``-inf`` and allowed positions are ``0.0``.
        """
        mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )
        return mask

    def _make_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create a boolean padding mask: ``True`` where the token is PAD.

        Shape: ``(B, T)``.
        """
        return input_ids == self.pad_id

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        sep_positions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the model on a batch of token sequences.

        Parameters
        ----------
        input_ids : Tensor, shape ``(B, T)``
            Integer token IDs.
        sep_positions : Tensor, shape ``(B,)``, optional
            Index of the [SEP] token in each sequence.  If ``None``, the
            position is auto-detected as the *first* occurrence of token ID 3
            (the default SEP ID) in each sequence.

        Returns
        -------
        logits : Tensor, shape ``(B, T, vocab_size)``
        value : Tensor, shape ``(B,)``
            Value estimate in [0, 1].
        """
        B, T = input_ids.shape
        device = input_ids.device

        # ---- Embeddings ----
        positions = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.embed_dropout(x)

        # ---- Masks ----
        causal_mask = self._make_causal_mask(T, device)           # (T, T)
        padding_mask = self._make_padding_mask(input_ids)         # (B, T)

        # ---- Dummy memory (decoder-only: no encoder output) ----
        # Shape: (B, 1, d_model) — a single dummy token of zeros
        dummy_memory = torch.zeros(B, 1, self.d_model, device=device)

        # ---- Transformer Decoder ----
        # tgt_mask: causal mask for self-attention (T, T)
        # tgt_key_padding_mask: padding mask (B, T)
        # memory_key_padding_mask: not needed (dummy is never padded)
        hidden = self.transformer_decoder(
            tgt=x,
            memory=dummy_memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )

        # ---- Output logits ----
        hidden_normed = self.output_norm(hidden)
        logits = self.output_projection(hidden_normed)  # (B, T, vocab_size)

        # ---- Value head ----
        if sep_positions is None:
            # Auto-detect: first occurrence of SEP (id=3) per sequence
            sep_id = 3
            # (B, T) == sep_id -> bool; argmax gives first True index
            sep_positions = (input_ids == sep_id).long().argmax(dim=1)

        # Gather hidden state at [SEP] positions
        # sep_positions: (B,) -> (B, 1, d_model) index
        sep_idx = sep_positions.to(device).unsqueeze(1).unsqueeze(2)
        sep_idx = sep_idx.expand(-1, -1, self.d_model)  # (B, 1, d_model)
        sep_hidden = hidden_normed.gather(1, sep_idx).squeeze(1)  # (B, d_model)

        value = torch.sigmoid(self.value_head(sep_hidden)).squeeze(-1)  # (B,)

        return logits, value

    # ------------------------------------------------------------------
    # Autoregressive generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 50,
        vocab: object = None,
        temperature: float = 1.0,
        eos_id: int = 2,
    ) -> torch.Tensor:
        """Autoregressively generate tokens given a prompt.

        Parameters
        ----------
        prompt : Tensor, shape ``(1, T_prompt)``
            Starting token IDs (should include [BOS] ... [SEP]).
        max_new_tokens : int
            Maximum number of tokens to generate after the prompt.
        vocab : Vocabulary, optional
            Vocabulary object, used by ``_apply_arity_mask`` (placeholder).
        temperature : float
            Sampling temperature.  Lower = more greedy.
        eos_id : int
            Token ID for [EOS].  Generation stops when this is produced.

        Returns
        -------
        Tensor, shape ``(1, T_prompt + generated)``
            The full sequence including prompt and generated tokens.
        """
        self.eval()
        device = next(self.parameters()).device
        generated = prompt.to(device)

        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if necessary
            input_ids = generated[:, -self.max_seq_len :]

            logits, _ = self.forward(input_ids)
            # Take logits at the last position
            next_logits = logits[:, -1, :]  # (1, vocab_size)

            # Apply arity mask (placeholder — will be filled in Task 8)
            next_logits = self._apply_arity_mask(next_logits, generated, vocab)

            # Temperature scaling
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == eos_id:
                break

        return generated

    def _apply_arity_mask(
        self,
        logits: torch.Tensor,
        generated_so_far: torch.Tensor,
        vocab: object,
    ) -> torch.Tensor:
        """Apply arity-constrained masking to logits during generation.

        Extracts the tokens generated after the last SEP in
        ``generated_so_far``, computes the set of allowed next tokens
        via :class:`ArityMask`, and sets disallowed logits to ``-inf``.

        Parameters
        ----------
        logits : Tensor, shape ``(1, vocab_size)``
        generated_so_far : Tensor, shape ``(1, T)``
        vocab : Vocabulary
            Must be a :class:`~integrate_zero.data.vocabulary.Vocabulary`
            instance.  If ``None``, the mask is skipped and logits are
            returned unchanged.

        Returns
        -------
        Tensor, shape ``(1, vocab_size)``
        """
        if vocab is None:
            return logits

        # Build the ArityMask (lightweight; could be cached but creation
        # is cheap enough for per-step use).
        arity_mask = ArityMask(vocab)

        # Extract token IDs after the last SEP in the generated sequence.
        token_ids = generated_so_far[0].tolist()
        sep_id = vocab.sep_id

        # Find last SEP position
        last_sep_idx = -1
        for i, tid in enumerate(token_ids):
            if tid == sep_id:
                last_sep_idx = i

        if last_sep_idx == -1:
            # No SEP found -- cannot apply mask meaningfully.
            return logits

        # Tokens after SEP (the output expression built so far)
        after_sep_ids = token_ids[last_sep_idx + 1 :]
        after_sep_tokens = []
        for tid in after_sep_ids:
            tok = vocab.id_to_token(tid)
            if tok is not None:
                after_sep_tokens.append(tok)

        # Get allowed token IDs
        allowed = arity_mask.get_allowed_tokens(after_sep_tokens)

        # Build mask: set disallowed positions to -inf
        mask = torch.full_like(logits, float("-inf"))
        allowed_list = list(allowed)
        if allowed_list:
            mask[0, allowed_list] = 0.0

        return logits + mask
