"""Neural-network definitions."""

import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 114,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        sequence_length: int = 4,
    ) -> None:
        super().__init__()
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.sequence_length = sequence_length
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(sequence_length, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.0,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def _embed(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.sequence_length:
            raise ValueError(
                f"expected inputs shaped (batch, {self.sequence_length}), got {tuple(inputs.shape)}"
            )
        return self.embed(inputs) + self.pos_embed.unsqueeze(0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.transformer(self._embed(inputs))
        return self.output(hidden[:, -1, :])

    def get_hidden_states(self, inputs: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Return the last-token representation after a selected encoder layer."""
        if not 0 <= layer_idx < self.n_layers:
            raise IndexError(f"layer_idx must be in [0, {self.n_layers})")
        hidden = self._embed(inputs)
        for index, layer in enumerate(self.transformer.layers):
            hidden = layer(hidden)
            if index == layer_idx:
                return hidden[:, -1, :].detach()
        raise RuntimeError("transformer layer was not reached")
