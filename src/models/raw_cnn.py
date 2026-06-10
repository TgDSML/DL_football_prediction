from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv(x)
        x = x + residual
        return self.activation(x)


class ConvBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv(x)
        x = x + residual
        return self.activation(x)


class Conv1DMatchPredictor(nn.Module):
    """
    1D CNN for football match sequences.

    Expected input:
        (batch_size, num_features, sequence_length)
    """

    def __init__(
        self,
        input_channels: int,
        num_classes: int = 3,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.3,
        classifier_hidden: int = 128,
    ) -> None:
        super().__init__()

        if channels is None:
            channels = [128, 128, 256]

        self.input_channels = input_channels
        self.num_classes = num_classes

        blocks = []
        in_channels = input_channels
        for out_channels in channels:
            blocks.append(
                ConvBlock1D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )
            blocks.append(nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True))
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        fused_dim = in_channels * 2
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, classifier_hidden),
            nn.BatchNorm1d(classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Conv1DMatchPredictor expected 3D input (N, F, k), got shape {tuple(x.shape)}")

        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Conv1DMatchPredictor expected {self.input_channels} channels, got {x.shape[1]}"
            )

        x = self.features(x)
        avg_feat = self.global_avg_pool(x).squeeze(-1)
        max_feat = self.global_max_pool(x).squeeze(-1)
        x = torch.cat([avg_feat, max_feat], dim=1)
        return self.classifier(x)


class Conv2DMatchPredictor(nn.Module):
    """
    2D CNN for football match sequences.

    Expected input:
        (batch_size, input_channels, input_height, input_width)

    Recommended:
        (batch_size, 1, sequence_length, num_features)
    """

    def __init__(
        self,
        input_height: int,
        input_width: int,
        input_channels: int = 1,
        num_classes: int = 3,
        channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.3,
        classifier_hidden: int = 128,
    ) -> None:
        super().__init__()

        if channels is None:
            channels = [32, 64, 128]

        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.num_classes = num_classes

        layers = []
        in_channels = input_channels

        for i, out_channels in enumerate(channels):
            layers.append(
                ConvBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
            )
            if i < len(channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), ceil_mode=True))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))

        fused_dim = in_channels * 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fused_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Conv2DMatchPredictor expected 4D input (N, C, H, W), got shape {tuple(x.shape)}")

        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Conv2DMatchPredictor expected {self.input_channels} channels, got {x.shape[1]}"
            )

        if x.shape[2] != self.input_height or x.shape[3] != self.input_width:
            raise ValueError(
                f"Conv2DMatchPredictor expected spatial size "
                f"({self.input_height}, {self.input_width}), got ({x.shape[2]}, {x.shape[3]})"
            )

        x = self.features(x)
        avg_feat = self.global_avg_pool(x)
        max_feat = self.global_max_pool(x)
        x = torch.cat([avg_feat, max_feat], dim=1)
        return self.classifier(x)


class HybridCNNMatchPredictor(nn.Module):
    """
    Hybrid CNN combining:
    - 1D temporal branch on (N, F, k)
    - 2D spatial-temporal branch on (N, 1, k, F)

    Expected input:
        (batch_size, sequence_length, num_features)
    """

    def __init__(
        self,
        input_channels: int,
        sequence_length: int,
        num_classes: int = 3,
        conv1d_channels: list[int] | None = None,
        conv2d_channels: list[int] | None = None,
        dropout: float = 0.3,
        classifier_hidden: int = 128,
    ) -> None:
        super().__init__()

        if conv1d_channels is None:
            conv1d_channels = [128, 256]
        if conv2d_channels is None:
            conv2d_channels = [32, 64]

        self.input_channels = input_channels
        self.sequence_length = sequence_length
        self.num_classes = num_classes

        branch1d = []
        in_1d = input_channels
        for out_1d in conv1d_channels:
            branch1d.append(ConvBlock1D(in_1d, out_1d, kernel_size=3, dropout=dropout))
            branch1d.append(nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True))
            in_1d = out_1d
        self.branch1d = nn.Sequential(*branch1d)
        self.pool1d_avg = nn.AdaptiveAvgPool1d(1)
        self.pool1d_max = nn.AdaptiveMaxPool1d(1)

        branch2d = []
        in_2d = 1
        for i, out_2d in enumerate(conv2d_channels):
            branch2d.append(ConvBlock2D(in_2d, out_2d, kernel_size=3, dropout=dropout))
            if i < len(conv2d_channels) - 1:
                branch2d.append(nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), ceil_mode=True))
            in_2d = out_2d
        self.branch2d = nn.Sequential(*branch2d)
        self.pool2d_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2d_max = nn.AdaptiveMaxPool2d((1, 1))

        fusion_dim = (conv1d_channels[-1] * 2) + (conv2d_channels[-1] * 2)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, classifier_hidden),
            nn.BatchNorm1d(classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"HybridCNNMatchPredictor expected 3D input (N, k, F), got shape {tuple(x.shape)}")

        if x.shape[1] != self.sequence_length:
            raise ValueError(
                f"HybridCNNMatchPredictor expected sequence length {self.sequence_length}, got {x.shape[1]}"
            )

        if x.shape[2] != self.input_channels:
            raise ValueError(
                f"HybridCNNMatchPredictor expected feature dimension {self.input_channels}, got {x.shape[2]}"
            )

        batch_size = x.size(0)

        x1d = x.transpose(1, 2).contiguous()
        x1d = self.branch1d(x1d)
        x1d_avg = self.pool1d_avg(x1d).view(batch_size, -1)
        x1d_max = self.pool1d_max(x1d).view(batch_size, -1)
        x1d = torch.cat([x1d_avg, x1d_max], dim=1)

        x2d = x.unsqueeze(1)
        x2d = self.branch2d(x2d)
        x2d_avg = self.pool2d_avg(x2d).view(batch_size, -1)
        x2d_max = self.pool2d_max(x2d).view(batch_size, -1)
        x2d = torch.cat([x2d_avg, x2d_max], dim=1)

        fused = torch.cat([x1d, x2d], dim=1)
        return self.classifier(fused)


def build_cnn_model(model_type: str = "conv1d", **kwargs) -> nn.Module:
    if model_type == "conv1d":
        return Conv1DMatchPredictor(**kwargs)
    if model_type == "conv2d":
        return Conv2DMatchPredictor(**kwargs)
    if model_type == "hybrid":
        return HybridCNNMatchPredictor(**kwargs)
    raise ValueError(f"Unknown model_type: {model_type}")