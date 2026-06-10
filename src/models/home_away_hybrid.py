from __future__ import annotations

import torch
from torch import nn


class HomeAwayHybridCNNMatchPredictor(nn.Module):
    """
    Hybrid CNN for football match prediction that separates home and away histories.

    Expected input:
        x.shape == (batch_size, sequence_length, num_features)

    The feature dimension must be even and interpreted as:
        [home_features | away_features]
    where the first half of channels describes home-team historical features
    and the second half describes away-team historical features.
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int = 3,
        conv1d_filters: list[int] | None = None,
        conv2d_filters: list[int] | None = None,
        hidden_size: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if input_channels % 2 != 0:
            raise ValueError("input_channels must be even for home/away splitting")

        self.input_channels = input_channels
        self.team_channels = input_channels // 2
        self.num_classes = num_classes

        if conv1d_filters is None:
            conv1d_filters = [64, 128]
        if conv2d_filters is None:
            conv2d_filters = [32, 64]

        self.home_branch = self._make_conv1d_branch(self.team_channels, conv1d_filters, dropout)
        self.away_branch = self._make_conv1d_branch(self.team_channels, conv1d_filters, dropout)

        self.cross_branch = self._make_conv2d_branch(1, conv2d_filters, dropout)

        self.home_pool = nn.AdaptiveAvgPool1d(1)
        self.away_pool = nn.AdaptiveAvgPool1d(1)
        self.cross_pool = nn.AdaptiveAvgPool2d((1, 1))

        fusion_dim = conv1d_filters[-1] * 2 + conv1d_filters[-1] + conv2d_filters[-1]
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    @staticmethod
    def _make_conv1d_branch(in_channels: int, filters: list[int], dropout: float) -> nn.Module:
        layers: list[nn.Module] = []
        for out_channels in filters:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        return nn.Sequential(*layers)

    @staticmethod
    def _make_conv2d_branch(in_channels: int, filters: list[int], dropout: float) -> nn.Module:
        layers: list[nn.Module] = []
        for out_channels in filters:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"HomeAwayHybridCNNMatchPredictor expected 3D input (N, k, F), got {tuple(x.shape)}"
            )

        if x.shape[2] != self.input_channels:
            raise ValueError(
                f"HomeAwayHybridCNNMatchPredictor expected {self.input_channels} features, got {x.shape[2]}"
            )

        home_x = x[:, :, : self.team_channels].transpose(1, 2)
        away_x = x[:, :, self.team_channels :].transpose(1, 2)

        home_x = self.home_branch(home_x)
        away_x = self.away_branch(away_x)

        home_x = self.home_pool(home_x).squeeze(-1)
        away_x = self.away_pool(away_x).squeeze(-1)

        diff_x = torch.abs(home_x - away_x)

        cross_x = x.unsqueeze(1)
        cross_x = self.cross_branch(cross_x)
        cross_x = self.cross_pool(cross_x).view(x.size(0), -1)

        fused = torch.cat([home_x, away_x, diff_x, cross_x], dim=1)
        logits = self.classifier(fused)
        return logits


def build_home_away_hybrid_cnn(
    input_channels: int,
    num_classes: int = 3,
    conv1d_filters: list[int] | None = None,
    conv2d_filters: list[int] | None = None,
    hidden_size: int = 128,
    dropout: float = 0.3,
) -> HomeAwayHybridCNNMatchPredictor:
    """Factory helper for the home/away hybrid CNN."""
    return HomeAwayHybridCNNMatchPredictor(
        input_channels=input_channels,
        num_classes=num_classes,
        conv1d_filters=conv1d_filters,
        conv2d_filters=conv2d_filters,
        hidden_size=hidden_size,
        dropout=dropout,
    )


if __name__ == "__main__":
    model = build_home_away_hybrid_cnn(input_channels=124, num_classes=3)
    print(model)
    x = torch.randn(2, 10, 124)
    out = model(x)
    print("output shape", out.shape)
