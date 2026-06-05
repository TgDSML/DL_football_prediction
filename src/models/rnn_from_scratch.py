from __future__ import annotations

import torch
from torch import nn


class ScratchRNNCell(nn.Module):
    """A minimal recurrent cell implemented with explicit affine transforms."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim, bias=False)
        self.hidden_to_hidden = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        return torch.tanh(
            self.input_to_hidden(x_t) + self.hidden_to_hidden(h_prev) + self.bias
        )


class ScratchRNNClassifier(nn.Module):
    """Vanilla RNN classifier using the final hidden state for prediction."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int = 3) -> None:
        super().__init__()
        self.encoder = ScratchRNNEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.rnn_cell = self.encoder.rnn_cell
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


class ScratchRNNEncoder(nn.Module):
    """Manual vanilla RNN encoder returning the final hidden state."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn_cell = ScratchRNNCell(input_dim=input_dim, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected input with shape [batch, sequence_length, input_dim], got {x.shape}"
            )

        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        for timestep in range(x.size(1)):
            h = self.rnn_cell(x[:, timestep, :], h)
        return h


class DualScratchRNNClassifier(nn.Module):
    """Two-stream scratch RNN classifier for separate home and away histories."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int = 3,
        dropout: float = 0.0,
        shared_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.shared_encoder = shared_encoder
        self.home_encoder = ScratchRNNEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        if shared_encoder:
            self.away_encoder = self.home_encoder
        else:
            self.away_encoder = ScratchRNNEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, home_seq: torch.Tensor, away_seq: torch.Tensor) -> torch.Tensor:
        h_home = self.home_encoder(home_seq)
        h_away = self.away_encoder(away_seq)
        h = torch.cat([h_home, h_away], dim=-1)
        return self.classifier(self.dropout(h))
