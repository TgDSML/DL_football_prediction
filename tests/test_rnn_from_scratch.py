from __future__ import annotations

import torch
from torch import nn

from src.models.rnn_from_scratch import (
    DualScratchRNNClassifier,
    ScratchRNNClassifier,
    ScratchRNNEncoder,
)


def test_scratch_rnn_forward_backward_without_builtin_recurrent_layers() -> None:
    model = ScratchRNNClassifier(input_dim=16, hidden_dim=32, num_classes=3)
    inputs = torch.randn(4, 5, 16)
    targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    logits = model(inputs)
    loss = nn.CrossEntropyLoss()(logits, targets)
    loss.backward()

    assert logits.shape == (4, 3)
    assert model.rnn_cell.input_to_hidden.weight.grad is not None
    assert model.classifier.weight.grad is not None

    forbidden_modules = (nn.RNN, nn.LSTM, nn.GRU)
    assert not any(isinstance(module, forbidden_modules) for module in model.modules())


def test_scratch_rnn_encoder_returns_final_hidden_state() -> None:
    encoder = ScratchRNNEncoder(input_dim=16, hidden_dim=32)
    inputs = torch.randn(4, 5, 16)

    hidden = encoder(inputs)

    assert hidden.shape == (4, 32)


def test_dual_scratch_rnn_forward_backward_without_builtin_recurrent_layers() -> None:
    model = DualScratchRNNClassifier(input_dim=16, hidden_dim=32, num_classes=3)
    home_seq = torch.randn(4, 5, 16)
    away_seq = torch.randn(4, 5, 16)
    targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    logits = model(home_seq, away_seq)
    loss = nn.CrossEntropyLoss()(logits, targets)
    loss.backward()

    assert logits.shape == (4, 3)
    assert model.home_encoder.rnn_cell.input_to_hidden.weight.grad is not None
    assert model.away_encoder.rnn_cell.input_to_hidden.weight.grad is not None
    assert model.classifier.weight.grad is not None

    forbidden_modules = (nn.RNN, nn.LSTM, nn.GRU)
    assert not any(isinstance(module, forbidden_modules) for module in model.modules())


def test_dual_scratch_rnn_shared_encoder_uses_same_object() -> None:
    model = DualScratchRNNClassifier(
        input_dim=16,
        hidden_dim=32,
        num_classes=3,
        shared_encoder=True,
    )

    assert model.home_encoder is model.away_encoder
