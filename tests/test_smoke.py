from __future__ import annotations

import numpy as np

from src.dataset import FootballSequenceDataset
from src.model import build_model


def test_training_entrypoint_imports() -> None:
    import src.train  # noqa: F401
    import src.pipelines.lstm_pipeline  # noqa: F401


def test_rnn_forward_pass() -> None:
    sequences = np.random.rand(4, 5, 16).astype("float32")
    labels = np.array([0, 1, 2, 0])
    dataset = FootballSequenceDataset(sequences, labels)
    model = build_model(
        "rnn",
        input_size=16,
        hidden_size=32,
        num_layers=1,
        dropout=0.1,
        num_classes=3,
    )

    output = model(dataset[0].sequences.unsqueeze(0))

    assert output.shape == (1, 3)


def test_lstm_forward_pass() -> None:
    sequences = np.random.rand(4, 5, 16).astype("float32")
    labels = np.array([0, 1, 2, 0])
    dataset = FootballSequenceDataset(sequences, labels)
    model = build_model(
        "lstm",
        input_size=16,
        hidden_size=32,
        num_layers=1,
        dropout=0.1,
        num_classes=3,
    )

    output = model(dataset[0].sequences.unsqueeze(0))

    assert output.shape == (1, 3)
