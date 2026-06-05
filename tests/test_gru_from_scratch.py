from __future__ import annotations

import torch
from torch import nn

from src.models.gru_from_scratch import (
    DualScratchGRUClassifier,
    ScratchGRUCell,
    ScratchGRUEncoder,
)


def test_scratch_gru_cell_forward_pass() -> None:
    cell = ScratchGRUCell(input_dim=16, hidden_dim=32)
    x_t = torch.randn(4, 16)
    h_prev = torch.randn(4, 32)

    h = cell(x_t, h_prev)

    assert h.shape == (4, 32)


def test_scratch_gru_encoder_forward_pass() -> None:
    encoder = ScratchGRUEncoder(input_dim=16, hidden_dim=32)
    inputs = torch.randn(4, 5, 16)

    hidden = encoder(inputs)

    assert hidden.shape == (4, 32)


def test_dual_scratch_gru_forward_backward_without_builtin_recurrent_layers() -> None:
    model = DualScratchGRUClassifier(input_dim=16, hidden_dim=32, num_classes=3)
    home_seq = torch.randn(4, 5, 16)
    away_seq = torch.randn(4, 5, 16)
    targets = torch.tensor([0, 1, 2, 0], dtype=torch.long)

    logits = model(home_seq, away_seq)
    loss = nn.CrossEntropyLoss()(logits, targets)
    loss.backward()

    assert logits.shape == (4, 3)
    assert model.home_encoder.gru_cell.input_to_update.weight.grad is not None
    assert model.away_encoder.gru_cell.input_to_candidate.weight.grad is not None
    assert model.classifier.weight.grad is not None

    forbidden_modules = (nn.GRU, nn.GRUCell, nn.RNN, nn.LSTM)
    assert not any(isinstance(module, forbidden_modules) for module in model.modules())


def test_dual_scratch_gru_shared_encoder_uses_same_object() -> None:
    model = DualScratchGRUClassifier(
        input_dim=16,
        hidden_dim=32,
        num_classes=3,
        shared_encoder=True,
    )

    assert model.home_encoder is model.away_encoder
