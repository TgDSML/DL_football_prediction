"""
Data loading utilities for CNN sequences.

Loads pre-processed sequences from sequence_engineering.py
and provides PyTorch DataLoaders.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class CNNSequenceDataset(Dataset):
    """
    PyTorch Dataset for CNN sequences.
    """

    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        metadata: pd.DataFrame | None = None,
    ) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.metadata = metadata.reset_index(drop=True) if metadata is not None else None

        if len(self.sequences) != len(self.labels):
            raise ValueError("sequences and labels must have same length")

        if self.metadata is not None and len(self.metadata) != len(self.labels):
            raise ValueError("metadata and labels must have same length")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {
            "sequences": self.sequences[index],
            "labels": self.labels[index],
            "index": index,
            "metadata": self.metadata.iloc[index].to_dict() if self.metadata is not None else None,
        }


class CNNDataModule:
    """
    Data loading orchestrator for CNN training.

    Expected structure from your current sequence engineering script:
    - NPZ files saved beside sequence_engineering.py:
        k_3_train_sequences.npz
        k_3_val_sequences.npz
        k_3_test_sequences.npz
        ...
    - Metadata saved under:
        artifacts/cnn/k_3/train_metadata.csv
        artifacts/cnn/k_3/val_metadata.csv
        artifacts/cnn/k_3/test_metadata.csv
        ...
    """

    def __init__(
        self,
        data_dir: str | Path,
        sequence_length: int,
        batch_size: int = 32,
        num_workers: int = 0,
        normalize: bool = True,
        use_cnn_format: bool = True,
        eps: float = 1e-8,
    ) -> None:
        """
        Args:
            data_dir: Directory where the NPZ files live, i.e. the script directory
            sequence_length: One of 3, 5, 10
            batch_size: Batch size for DataLoaders
            num_workers: Number of workers for DataLoader
            normalize: Whether to normalize sequences
            use_cnn_format: If True load 'sequences_cnn' with shape (N, F, k),
                            else load 'sequences' with shape (N, k, F)
            eps: Small constant to avoid divide-by-zero
        """
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.use_cnn_format = use_cnn_format
        self.eps = eps

        self.project_root = self.data_dir.parent.parent.parent
        self.metadata_dir = self.project_root / "artifacts" / "cnn" / f"k_{self.sequence_length}"

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.normalizer = None

    def _npz_path(self, split_name: str) -> Path:
        return self.data_dir / f"k_{self.sequence_length}_{split_name}_sequences.npz"

    def _metadata_path(self, split_name: str) -> Path:
        return self.metadata_dir / f"{split_name}_metadata.csv"

    def load_split(self, split_name: str = "train") -> CNNSequenceDataset:
        """
        Load one split using your current naming convention.
        """
        seq_file = self._npz_path(split_name)
        meta_file = self._metadata_path(split_name)

        if not seq_file.exists():
            raise FileNotFoundError(f"Sequence file not found: {seq_file}")

        data = np.load(seq_file)

        array_key = "sequences_cnn" if self.use_cnn_format else "sequences"
        if array_key not in data:
            raise KeyError(f"Expected key '{array_key}' in {seq_file}, found keys: {list(data.keys())}")

        sequences = data[array_key]
        labels = data["labels"]

        metadata = None
        if meta_file.exists():
            metadata = pd.read_csv(meta_file)

        return CNNSequenceDataset(sequences, labels, metadata)

    def setup(self):
        print(f"Loading CNN sequences for k={self.sequence_length}...")
        print(f"  NPZ dir: {self.data_dir}")
        print(f"  Metadata dir: {self.metadata_dir}")

        self.train_dataset = self.load_split("train")
        print(f"  Train: {len(self.train_dataset)} samples")

        try:
            self.val_dataset = self.load_split("val")
            print(f"  Val:   {len(self.val_dataset)} samples")
        except FileNotFoundError:
            self.val_dataset = None
            print("  Val:   Not found")

        try:
            self.test_dataset = self.load_split("test")
            print(f"  Test:  {len(self.test_dataset)} samples")
        except FileNotFoundError:
            self.test_dataset = None
            print("  Test:  Not found")

        if self.normalize:
            self._compute_normalizer()

    def _compute_normalizer(self):
        """
        Compute channel-wise mean/std from training data only.

        If use_cnn_format=True:
            sequences shape = (N, F, k)
            mean/std shape = (F,)
            reduce over dims (0, 2)

        If use_cnn_format=False:
            sequences shape = (N, k, F)
            mean/std shape = (F,)
            reduce over dims (0, 1)
        """
        sequences = self.train_dataset.sequences

        if self.use_cnn_format:
            mean = sequences.mean(dim=(0, 2))
            std = sequences.std(dim=(0, 2), unbiased=False)
        else:
            mean = sequences.mean(dim=(0, 1))
            std = sequences.std(dim=(0, 1), unbiased=False)

        std = torch.where(std < self.eps, torch.ones_like(std), std)

        self.normalizer = {
            "mean": mean,
            "std": std,
        }

        print(
            f"  Channel normalizer ready: "
            f"{mean.numel()} feature channels"
        )

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.normalizer is None:
            return tensor

        mean = self.normalizer["mean"]
        std = self.normalizer["std"]

        if self.use_cnn_format:
            mean = mean.view(1, -1, 1)
            std = std.view(1, -1, 1)
        else:
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)

        return (tensor - mean) / std

    def collate_fn(self, batch: list[dict]) -> dict:
        sequences = torch.stack([item["sequences"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])

        if self.normalize:
            sequences = self._normalize(sequences)

        return {
            "sequences": sequences,
            "labels": labels,
            "indices": [item["index"] for item in batch],
            "metadata": [item["metadata"] for item in batch],
        }

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() first")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self.val_dataset is None:
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader | None:
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )


def load_cnn_data(
    data_dir: str | Path,
    sequence_length: int,
    batch_size: int = 32,
    normalize: bool = True,
    use_cnn_format: bool = True,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None]:
    """
    Convenience function to load CNN data.

    Args:
        data_dir: Directory containing files like k_5_train_sequences.npz
        sequence_length: One of 3, 5, 10
        batch_size: Batch size
        normalize: Whether to normalize
        use_cnn_format: True -> load sequences_cnn (N, F, k),
                        False -> load sequences (N, k, F)
        num_workers: DataLoader workers

    Returns:
        (train_loader, val_loader, test_loader)
    """
    module = CNNDataModule(
        data_dir=data_dir,
        sequence_length=sequence_length,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=normalize,
        use_cnn_format=use_cnn_format,
    )
    module.setup()

    return (
        module.train_dataloader(),
        module.val_dataloader(),
        module.test_dataloader(),
    )