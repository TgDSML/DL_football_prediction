from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class MatchBatch:
    sequences: torch.Tensor
    labels: torch.Tensor


class FootballSequenceDataset(Dataset[MatchBatch]):
    """Dataset for team-form sequences and match outcome labels."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        if len(sequences) != len(labels):
            raise ValueError("sequences and labels must have the same length")
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> MatchBatch:
        return MatchBatch(sequences=self.sequences[index], labels=self.labels[index])
