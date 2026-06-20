from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class CNNSequenceDataset(Dataset):
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
    Loader for leakage-safe sequence files produced by scripts/build_sequences.py.

    Expected files:
      home_away_seq10_train.npz
      home_away_seq10_val.npz
      home_away_seq10_test.npz

    NPZ keys:
      - X: (N, k, F)
      - y: (N,)
      - feature_names

    Metadata files:
      - home_away_seq10_train_metadata.csv
      - home_away_seq10_val_metadata.csv
      - home_away_seq10_test_metadata.csv
    """

    def __init__(
        self,
        data_dir: str | Path,
        sequence_length: int,
        variant: str = "home_away",
        batch_size: int = 32,
        num_workers: int = 0,
        normalize: bool = True,
        use_cnn_format: bool = True,
        eps: float = 1e-8,
        load_metadata: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.variant = variant
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.use_cnn_format = use_cnn_format
        self.eps = eps
        self.load_metadata = load_metadata

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.normalizer = None
        self.feature_names = None

    @property
    def file_stem(self) -> str:
        if self.sequence_length == 5:
            return self.variant
        return f"{self.variant}_seq{self.sequence_length}"

    def _npz_path(self, split_name: str) -> Path:
        return self.data_dir / f"{self.file_stem}_{split_name}.npz"

    def _metadata_path(self, split_name: str) -> Path:
        return self.data_dir / f"{self.file_stem}_{split_name}_metadata.csv"

    def load_split(self, split_name: str = "train") -> CNNSequenceDataset:
        seq_file = self._npz_path(split_name)
        meta_file = self._metadata_path(split_name)

        if not seq_file.exists():
            raise FileNotFoundError(f"Sequence file not found: {seq_file}")

        with np.load(seq_file, allow_pickle=False) as data:
            expected_keys = {"X", "y"}
            if not expected_keys.issubset(set(data.files)):
                raise KeyError(
                    f"Expected keys {expected_keys} in {seq_file}, found {list(data.files)}"
                )

            sequences = data["X"]  # (N, k, F)
            labels = data["y"]

            if "feature_names" in data and self.feature_names is None:
                self.feature_names = [str(x) for x in data["feature_names"]]

        if self.use_cnn_format:
            sequences = np.transpose(sequences, (0, 2, 1))  # (N, F, k)

        metadata = None
        if self.load_metadata and meta_file.exists():
            metadata = pd.read_csv(meta_file)

        return CNNSequenceDataset(sequences=sequences, labels=labels, metadata=metadata)

    def setup(self) -> None:
        print(f"Loading CNN sequences for k={self.sequence_length}...")
        print(f"  NPZ dir: {self.data_dir}")
        print(f"  Variant: {self.variant}")
        print(f"  CNN format: {self.use_cnn_format}")

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

    def _compute_normalizer(self) -> None:
        if self.train_dataset is None:
            raise RuntimeError("Train dataset must be loaded before normalization")

        sequences = self.train_dataset.sequences

        if self.use_cnn_format:
            mean = sequences.mean(dim=(0, 2))
            std = sequences.std(dim=(0, 2), unbiased=False)
        else:
            mean = sequences.mean(dim=(0, 1))
            std = sequences.std(dim=(0, 1), unbiased=False)

        std = torch.where(std < self.eps, torch.ones_like(std), std)
        self.normalizer = {"mean": mean, "std": std}

        print(f"  Channel normalizer ready: {mean.numel()} feature channels")

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
    variant: str = "home_away",
    batch_size: int = 32,
    normalize: bool = True,
    use_cnn_format: bool = True,
    num_workers: int = 0,
    load_metadata: bool = True,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None, CNNDataModule]:
    module = CNNDataModule(
        data_dir=data_dir,
        sequence_length=sequence_length,
        variant=variant,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=normalize,
        use_cnn_format=use_cnn_format,
        load_metadata=load_metadata,
    )
    module.setup()

    return (
        module.train_dataloader(),
        module.val_dataloader(),
        module.test_dataloader(),
        module,
    )


if __name__ == "__main__":
    data_dir = Path("data/processed/sequences")

    train_loader, val_loader, test_loader, module = load_cnn_data(
        data_dir=data_dir,
        sequence_length=10,
        variant="home_away",
        batch_size=32,
        normalize=True,
        use_cnn_format=True,
        num_workers=0,
        load_metadata=True,
    )

    first_batch = next(iter(train_loader))
    print("Batch sequence shape:", first_batch["sequences"].shape)
    print("Batch label shape:", first_batch["labels"].shape)
    print("Feature count:", len(module.feature_names) if module.feature_names is not None else "unknown")
