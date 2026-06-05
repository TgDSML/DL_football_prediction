from __future__ import annotations

import torch
from torch import nn


class ScratchGRUCell(nn.Module):
    """A GRU cell implemented with explicit gates and autograd tensors."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_to_update = nn.Linear(input_dim, hidden_dim, bias=False)
        self.hidden_to_update = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.update_bias = nn.Parameter(torch.zeros(hidden_dim))

        self.input_to_reset = nn.Linear(input_dim, hidden_dim, bias=False)
        self.hidden_to_reset = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.reset_bias = nn.Parameter(torch.zeros(hidden_dim))

        self.input_to_candidate = nn.Linear(input_dim, hidden_dim, bias=False)
        self.hidden_to_candidate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.candidate_bias = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        z_t = torch.sigmoid(
            self.input_to_update(x_t)
            + self.hidden_to_update(h_prev)
            + self.update_bias
        )
        r_t = torch.sigmoid(
            self.input_to_reset(x_t)
            + self.hidden_to_reset(h_prev)
            + self.reset_bias
        )
        h_candidate = torch.tanh(
            self.input_to_candidate(x_t)
            + self.hidden_to_candidate(r_t * h_prev)
            + self.candidate_bias
        )
        return (1.0 - z_t) * h_prev + z_t * h_candidate


class ScratchGRUEncoder(nn.Module):
    """Manual GRU encoder returning the final hidden state."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru_cell = ScratchGRUCell(input_dim=input_dim, hidden_dim=hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected input with shape [batch, sequence_length, input_dim], got {x.shape}"
            )

        batch_size = x.size(0)
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        for timestep in range(x.size(1)):
            h = self.gru_cell(x[:, timestep, :], h)
        return h


class DualScratchGRUClassifier(nn.Module):
    """Two-stream scratch GRU classifier for home and away histories."""

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
        self.home_encoder = ScratchGRUEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        if shared_encoder:
            self.away_encoder = self.home_encoder
        else:
            self.away_encoder = ScratchGRUEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, home_seq: torch.Tensor, away_seq: torch.Tensor) -> torch.Tensor:
        h_home = self.home_encoder(home_seq)
        h_away = self.away_encoder(away_seq)
        h = torch.cat([h_home, h_away], dim=-1)
        return self.classifier(self.dropout(h))
