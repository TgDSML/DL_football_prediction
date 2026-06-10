from __future__ import annotations

import torch
import torch.nn as nn


class Conv1DBlock(nn.Module):
    """A simple Conv1D block with BatchNorm, ReLU, and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNLSTMMatchPredictor(nn.Module):
    """Conv1D + LSTM classifier for sequence-based football match outcomes."""

    def __init__(
        self,
        input_features: int,
        num_classes: int = 3,
        conv_channels: list[int] | None = None,
        kernel_size: int = 3,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        if input_features <= 0:
            raise ValueError("input_features must be positive")
        if lstm_layers <= 0:
            raise ValueError("lstm_layers must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self.input_features = input_features
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.lstm_hidden_size = lstm_hidden_size

        if conv_channels is None:
            conv_channels = [64, 128]
        if len(conv_channels) == 0:
            raise ValueError("conv_channels must contain at least one channel size")

        blocks: list[nn.Module] = []
        in_channels = input_features
        for out_channels in conv_channels:
            blocks.append(
                Conv1DBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*blocks)
        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        classifier_input_features = lstm_hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_features, classifier_input_features),
            nn.BatchNorm1d(classifier_input_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_input_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: tensor of shape (batch_size, sequence_length, num_features)

        Returns:
            logits tensor of shape (batch_size, num_classes)
        """
        if x.ndim != 3:
            raise ValueError(
                f"CNNLSTMMatchPredictor expected 3D input (batch, sequence_length, num_features), got {tuple(x.shape)}"
            )
        if x.shape[2] != self.input_features:
            raise ValueError(
                f"CNNLSTMMatchPredictor expected {self.input_features} features, got {x.shape[2]}"
            )
        if x.shape[0] == 0:
            raise ValueError("CNNLSTMMatchPredictor batch size must be > 0")
        if x.shape[1] == 0:
            raise ValueError("CNNLSTMMatchPredictor sequence length must be > 0")

        x = x.transpose(1, 2)
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)

        lstm_output, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            hidden = h_n[-1]

        logits = self.classifier(hidden)
        return logits
