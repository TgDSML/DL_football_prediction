from __future__ import annotations

import torch
from torch import nn


class LSTMMatchPredictor(nn.Module):
    """LSTM baseline for pre-match form sequences."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.0,
        bidirectional: bool = False,
        classifier_hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
        )

        classifier_hidden_size = classifier_hidden_size or hidden_size
        classifier_input_size = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_input_size, classifier_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_size, num_classes),
        )

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(sequences)
        if self.bidirectional:
            # Concatenate last forward and backward hidden states
            final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            final_hidden = hidden[-1]
        return self.classifier(final_hidden)


class TransformerMatchPredictor(nn.Module):
    """Transformer encoder placeholder for future sequence experiments."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.input_projection(sequences))
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)


def build_model(model_type: str, **kwargs: int | float) -> nn.Module:
    if model_type == "lstm":
        return LSTMMatchPredictor(**kwargs)
    if model_type == "transformer":
        return TransformerMatchPredictor(**kwargs)
    raise ValueError(f"Unsupported model_type: {model_type}")
