# Improved CNN+LSTM baseline:
# - uses residual Conv1D blocks instead of single conv layers
# - applies Dropout1d for better channel-level regularization
# - adds optional MaxPool1d to reduce noise and shorten the sequence
# - summarizes LSTM output with mean + max pooling for richer sequence representation
# - uses LayerNorm before the classifier for more stable training
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualConv1DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout1d(dropout)

        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + residual
        x = self.relu(x)
        return x


class CNNLSTMMatchPredictor(nn.Module):
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
        use_pooling_summary: bool = True,
    ) -> None:
        super().__init__()

        if input_features <= 0:
            raise ValueError("input_features must be positive")
        if lstm_layers <= 0:
            raise ValueError("lstm_layers must be positive")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        if conv_channels is None:
            conv_channels = [64, 128]
        if not conv_channels:
            raise ValueError("conv_channels must contain at least one channel size")

        self.input_features = input_features
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.use_pooling_summary = use_pooling_summary

        blocks: list[nn.Module] = []
        in_channels = input_features
        for i, out_channels in enumerate(conv_channels):
            blocks.append(
                ResidualConv1DBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )
            if i < len(conv_channels) - 1:
                blocks.append(nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True))
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

        lstm_dim = lstm_hidden_size * (2 if bidirectional else 1)

        if use_pooling_summary:
            classifier_input = lstm_dim * 2
        else:
            classifier_input = lstm_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input),
            nn.Linear(classifier_input, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        x = x.transpose(1, 2).contiguous()
        x = self.feature_extractor(x)
        x = x.transpose(1, 2).contiguous()

        lstm_output, (h_n, _) = self.lstm(x)

        if self.use_pooling_summary:
            mean_pool = lstm_output.mean(dim=1)
            max_pool, _ = lstm_output.max(dim=1)
            features = torch.cat([mean_pool, max_pool], dim=1)
        else:
            if self.bidirectional:
                features = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                features = h_n[-1]

        logits = self.classifier(features)
        return logits