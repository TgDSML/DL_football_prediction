from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.dataset import FootballSequenceDataset
from src.model import build_model


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    sequences = np.random.rand(8, 5, 16).astype("float32")
    labels = np.random.randint(0, 3, size=8)

    dataset = FootballSequenceDataset(sequences, labels)
    model = build_model("lstm", **config.get("model"))
    logits = model(dataset[0].sequences.unsqueeze(0))

    assert len(dataset) == 8
    assert logits.shape == torch.Size([1, 3])
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
