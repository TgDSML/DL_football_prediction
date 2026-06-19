"""
CNN models for football match prediction.

Expected alignment with dataset_cnn.py:

- 1D CNN:
    load use_cnn_format=True
    input shape = (batch_size, num_features, sequence_length)

- 2D CNN:
    start from raw sequences with shape (batch_size, sequence_length, num_features)
    or reshape externally to (batch_size, 1, sequence_length, num_features)

- Hybrid CNN:
    expects raw sequence shape (batch_size, sequence_length, num_features)
    and internally creates:
        1D branch input: (batch_size, num_features, sequence_length)
        2D branch input: (batch_size, 1, sequence_length, num_features)
"""

import torch
import torch.nn as nn


class Conv1DMatchPredictor(nn.Module):
    """
    1D CNN for temporal football sequences.

    Expected input:
        x.shape == (batch_size, num_features, sequence_length)

    This matches sequences_cnn produced by the loader when use_cnn_format=True.
    """

    def __init__(
        self,
        input_channels: int = 124,
        num_classes: int = 3,
        num_filters: int | list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if num_filters is None:
            num_filters = [64, 128, 64]
        elif isinstance(num_filters, int):
            num_filters = [num_filters] * 3

        self.input_channels = input_channels
        self.num_classes = num_classes

        self.conv_layers = nn.ModuleList()
        in_channels = input_channels

        for out_channels in num_filters:
            block = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.conv_layers.append(block)
            in_channels = out_channels

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_features, sequence_length)

        Returns:
            logits: (batch_size, num_classes)
        """
        if x.ndim != 3:
            raise ValueError(f"Conv1DMatchPredictor expected 3D input (N, F, k), got shape {tuple(x.shape)}")

        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Conv1DMatchPredictor expected {self.input_channels} channels, got {x.shape[1]}"
            )

        for conv in self.conv_layers:
            x = conv(x)

        x = self.global_pool(x)
        x = x.squeeze(-1)
        logits = self.classifier(x)
        return logits


class Conv2DMatchPredictor(nn.Module):
    """
    2D CNN for football match prediction.

    Expected input:
        x.shape == (batch_size, input_channels, input_height, input_width)

    Recommended use:
        reshape raw sequences externally to:
            (batch_size, 1, sequence_length, num_features)
    so:
        input_channels = 1
        input_height = sequence_length
        input_width = num_features
    """

    def __init__(
        self,
        input_height: int,
        input_width: int,
        input_channels: int = 1,
        num_classes: int = 3,
        num_filters: int | list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if num_filters is None:
            num_filters = [32, 64, 128]
        elif isinstance(num_filters, int):
            num_filters = [num_filters] * 3

        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(input_channels, num_filters[0], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(num_filters[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2), stride=(1, 2)),

            nn.Conv2d(num_filters[0], num_filters[1], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2), stride=(1, 2)),

            nn.Conv2d(num_filters[1], num_filters[2], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_filters[-1], 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_channels, input_height, input_width)
        """
        if x.ndim != 4:
            raise ValueError(f"Conv2DMatchPredictor expected 4D input (N, C, H, W), got shape {tuple(x.shape)}")

        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Conv2DMatchPredictor expected {self.input_channels} channels, got {x.shape[1]}"
            )

        x = self.features(x)
        logits = self.classifier(x)
        return logits


class HybridCNNMatchPredictor(nn.Module):
    """
    Hybrid CNN combining:
    - 1D temporal branch on (N, F, k)
    - 2D spatial-temporal branch on (N, 1, k, F)

    Expected input:
        x.shape == (batch_size, sequence_length, num_features)
    """

    def __init__(
        self,
        input_channels: int = 124,
        num_classes: int = 3,
        conv1d_filters: list[int] | None = None,
        conv2d_filters: list[int] | None = None,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if conv1d_filters is None:
            conv1d_filters = [64, 128]
        if conv2d_filters is None:
            conv2d_filters = [32, 64]

        self.input_channels = input_channels
        self.num_classes = num_classes

        self.conv1d_layers = nn.ModuleList()
        in_channels = input_channels
        for out_channels in conv1d_filters:
            block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.conv1d_layers.append(block)
            in_channels = out_channels
        self.global_pool_1d = nn.AdaptiveAvgPool1d(1)

        self.conv2d_layers = nn.ModuleList()
        in_ch = 1
        for out_channels in conv2d_filters:
            block = nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.conv2d_layers.append(block)
            in_ch = out_channels
        self.global_pool_2d = nn.AdaptiveAvgPool2d((1, 1))

        fusion_size = conv1d_filters[-1] + conv2d_filters[-1]
        self.classifier = nn.Sequential(
            nn.Linear(fusion_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length, num_features)
        """
        if x.ndim != 3:
            raise ValueError(f"HybridCNNMatchPredictor expected 3D input (N, k, F), got shape {tuple(x.shape)}")

        if x.shape[2] != self.input_channels:
            raise ValueError(
                f"HybridCNNMatchPredictor expected feature dimension {self.input_channels}, got {x.shape[2]}"
            )

        batch_size = x.size(0)

        x1d = x.transpose(1, 2).contiguous()
        for conv in self.conv1d_layers:
            x1d = conv(x1d)
        x1d = self.global_pool_1d(x1d).view(batch_size, -1)

        x2d = x.unsqueeze(1)
        for conv in self.conv2d_layers:
            x2d = conv(x2d)
        x2d = self.global_pool_2d(x2d).view(batch_size, -1)

        fused = torch.cat([x1d, x2d], dim=1)
        logits = self.classifier(fused)
        return logits


def build_cnn_model(model_type: str = "conv1d", **kwargs) -> nn.Module:
    """
    Factory function for CNN models.

    Recommended alignment:
    - model_type='conv1d'  -> loader with use_cnn_format=True
    - model_type='conv2d'  -> loader with use_cnn_format=False, then reshape in training step
    - model_type='hybrid'  -> loader with use_cnn_format=False
    """
    if model_type == "conv1d":
        return Conv1DMatchPredictor(**kwargs)
    elif model_type == "conv2d":
        return Conv2DMatchPredictor(**kwargs)
    elif model_type == "hybrid":
        return HybridCNNMatchPredictor(**kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")