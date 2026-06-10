from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import copy
import json
import math
import random
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.raw_dataset_cnn import CNNDataModule
from src.models.cnn_lstm_2 import CNNLSTMMatchPredictor


SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = Path("data/processed/sequences_leakage_safe")
OUTPUT_DIR = Path("artifacts") / "cnn_lstm_2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANT = "home_away"
MODEL_TYPE = "cnnlstm"
SEQUENCE_LENGTH_CANDIDATES = [3, 5, 10]
NUM_CLASSES = 3
NUM_WORKERS = 0

MAX_EPOCHS = 40
PATIENCE = 8

LEARNING_RATE = 3e-4
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-3
LABEL_SMOOTHING = 0.0

CONV_CHANNELS = [64, 128]
KERNEL_SIZE = 3
LSTM_HIDDEN_SIZE = 128
LSTM_LAYERS = 1
DROPOUT = 0.3
BIDIRECTIONAL = False
USE_POOLING_SUMMARY = True

CLASS_NAMES = ["HomeWin", "Draw", "AwayWin"]


@dataclass
class RunResult:
    run_id: str
    model_type: str
    sequence_length: int
    learning_rate: float
    batch_size: int
    dropout: float
    lstm_hidden_size: int
    lstm_layers: int
    bidirectional: bool
    best_epoch: int
    val_loss: float
    val_accuracy: float
    val_f1_macro: float
    val_f1_weighted: float
    test_loss: float | None = None
    test_accuracy: float | None = None
    test_f1_macro: float | None = None
    test_f1_weighted: float | None = None
    report_path: str | None = None


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_data_module(sequence_length: int, batch_size: int) -> CNNDataModule:
    return CNNDataModule(
        data_dir=DATA_DIR,
        sequence_length=sequence_length,
        variant=VARIANT,
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
        normalize=True,
        use_cnn_format=False,
        load_metadata=True,
    )


def build_model(num_features: int) -> nn.Module:
    return CNNLSTMMatchPredictor(
        input_features=num_features,
        num_classes=NUM_CLASSES,
        conv_channels=CONV_CHANNELS,
        kernel_size=KERNEL_SIZE,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
        use_pooling_summary=USE_POOLING_SUMMARY,
    )


def prepare_inputs(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"CNNLSTM expects input of shape (N, k, F), got {tuple(x.shape)}")
    return x


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> dict:
    training = optimizer is not None
    model.train(training)

    losses: list[float] = []
    all_preds: list[int] = []
    all_targets: list[int] = []

    for batch in loader:
        x = batch["sequences"].to(DEVICE)
        y = batch["labels"].to(DEVICE)

        x = prepare_inputs(x)

        logits = model(x)
        loss = criterion(logits, y)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        preds = logits.argmax(dim=1)

        losses.append(loss.detach().item())
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(y.detach().cpu().tolist())

    return {
        "loss": float(np.mean(losses)) if losses else math.nan,
        "accuracy": accuracy_score(all_targets, all_preds),
        "f1_macro": f1_score(all_targets, all_preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(all_targets, all_preds, average="weighted", zero_division=0),
        "y_true": all_targets,
        "y_pred": all_preds,
    }


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader | None,
    criterion: nn.Module,
) -> dict | None:
    if loader is None:
        return None
    model.eval()
    with torch.no_grad():
        return run_epoch(model, loader, criterion, optimizer=None)


def is_better(candidate: dict, best_f1: float, best_acc: float, best_loss: float) -> bool:
    return (
        candidate["f1_macro"] > best_f1 + 1e-6
        or (
            abs(candidate["f1_macro"] - best_f1) <= 1e-6
            and candidate["accuracy"] > best_acc + 1e-6
        )
        or (
            abs(candidate["f1_macro"] - best_f1) <= 1e-6
            and abs(candidate["accuracy"] - best_acc) <= 1e-6
            and candidate["loss"] < best_loss
        )
    )


def write_run_report(
    result: RunResult,
    num_features: int,
    val_metrics: dict,
    test_metrics: dict | None,
) -> Path:
    report_path = OUTPUT_DIR / f"{result.run_id}_report.txt"

    lines = [
        f"Run ID: {result.run_id}",
        f"Model type: {result.model_type}",
        f"Sequence length: {result.sequence_length}",
        f"Num features: {num_features}",
        f"Learning rate: {result.learning_rate}",
        f"Batch size: {result.batch_size}",
        f"Dropout: {result.dropout}",
        f"LSTM hidden size: {result.lstm_hidden_size}",
        f"LSTM layers: {result.lstm_layers}",
        f"Bidirectional: {result.bidirectional}",
        f"Best epoch: {result.best_epoch}",
        "",
        "Selection rule",
        "--------------",
        "1. Validation Macro F1",
        "2. Validation Accuracy",
        "3. Validation Loss",
        "",
        "Validation metrics",
        "------------------",
        f"Loss: {val_metrics['loss']:.6f}",
        f"Accuracy: {val_metrics['accuracy']:.6f}",
        f"Macro F1: {val_metrics['f1_macro']:.6f}",
        f"Weighted F1: {val_metrics['f1_weighted']:.6f}",
        "",
        "Validation classification report",
        "--------------------------------",
        classification_report(
            val_metrics["y_true"],
            val_metrics["y_pred"],
            labels=[0, 1, 2],
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        ),
        "",
        "Validation confusion matrix",
        "---------------------------",
        str(confusion_matrix(val_metrics["y_true"], val_metrics["y_pred"], labels=[0, 1, 2])),
    ]

    if test_metrics is not None:
        lines.extend(
            [
                "",
                "Test metrics",
                "------------",
                f"Loss: {test_metrics['loss']:.6f}",
                f"Accuracy: {test_metrics['accuracy']:.6f}",
                f"Macro F1: {test_metrics['f1_macro']:.6f}",
                f"Weighted F1: {test_metrics['f1_weighted']:.6f}",
                "",
                "Test classification report",
                "--------------------------",
                classification_report(
                    test_metrics["y_true"],
                    test_metrics["y_pred"],
                    labels=[0, 1, 2],
                    target_names=CLASS_NAMES,
                    digits=4,
                    zero_division=0,
                ),
                "",
                "Test confusion matrix",
                "---------------------",
                str(confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"], labels=[0, 1, 2])),
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def fit_single_run(sequence_length: int) -> RunResult:
    run_id = f"cnnlstm_seq{sequence_length}_lr{LEARNING_RATE:g}_bs{BATCH_SIZE}"

    data_module = build_data_module(sequence_length=sequence_length, batch_size=BATCH_SIZE)
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    if val_loader is None:
        raise RuntimeError("Validation loader is required.")
    if data_module.feature_names is None:
        raise RuntimeError("feature_names were not loaded from the NPZ files.")

    num_features = len(data_module.feature_names)

    model = build_model(num_features=num_features).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_state = None
    best_epoch = -1
    best_val_macro_f1 = -1.0
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_val_metrics = None
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer)

        model.eval()
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, optimizer=None)

        if is_better(val_metrics, best_val_macro_f1, best_val_accuracy, best_val_loss):
            best_val_macro_f1 = val_metrics["f1_macro"]
            best_val_accuracy = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_val_metrics = copy.deepcopy(val_metrics)
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"[{run_id}] epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"train_f1_macro={train_metrics['f1_macro']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1_macro={val_metrics['f1_macro']:.4f}"
        )

        if patience_counter >= PATIENCE:
            print(f"[{run_id}] early stopping at epoch {epoch}")
            break

    if best_state is None or best_val_metrics is None:
        raise RuntimeError(f"No best state recorded for {run_id}")

    model.load_state_dict(best_state)
    test_metrics = evaluate_loader(model, test_loader, criterion)

    result = RunResult(
        run_id=run_id,
        model_type=MODEL_TYPE,
        sequence_length=sequence_length,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        dropout=DROPOUT,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS,
        bidirectional=BIDIRECTIONAL,
        best_epoch=best_epoch,
        val_loss=best_val_loss,
        val_accuracy=best_val_accuracy,
        val_f1_macro=best_val_macro_f1,
        val_f1_weighted=best_val_metrics["f1_weighted"],
        test_loss=None if test_metrics is None else test_metrics["loss"],
        test_accuracy=None if test_metrics is None else test_metrics["accuracy"],
        test_f1_macro=None if test_metrics is None else test_metrics["f1_macro"],
        test_f1_weighted=None if test_metrics is None else test_metrics["f1_weighted"],
    )

    report_path = write_run_report(
        result=result,
        num_features=num_features,
        val_metrics=best_val_metrics,
        test_metrics=test_metrics,
    )
    result.report_path = str(report_path)
    return result


def main() -> None:
    set_seed(SEED)

    all_results: list[dict] = []

    print("\n=== CNN+LSTM sequence length sweep ===")
    for sequence_length in SEQUENCE_LENGTH_CANDIDATES:
        result = fit_single_run(sequence_length=sequence_length)
        all_results.append(asdict(result))

    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)

    best_by_group = df.sort_values(
        ["val_f1_macro", "val_accuracy", "val_loss"],
        ascending=[False, False, True],
    ).drop_duplicates(subset=["sequence_length"])
    best_by_group.to_csv(OUTPUT_DIR / "best_by_group.csv", index=False)

    overall_best = df.sort_values(
        ["val_f1_macro", "val_accuracy", "val_loss"],
        ascending=[False, False, True],
    ).iloc[0]

    (OUTPUT_DIR / "overall_best.json").write_text(
        json.dumps(
            {
                k: (v.item() if isinstance(v, (np.integer, np.floating)) else v)
                for k, v in overall_best.to_dict().items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_lines = [
        "CNN+LSTM training summary",
        "=========================",
        "",
        "Selection rule:",
        "1. Validation Macro F1",
        "2. Validation Accuracy",
        "3. Validation Loss",
        "",
        f"Model type: {MODEL_TYPE}",
        f"Variant: {VARIANT}",
        f"Sequence lengths tried: {SEQUENCE_LENGTH_CANDIDATES}",
        f"Learning rate: {LEARNING_RATE}",
        f"Batch size: {BATCH_SIZE}",
        f"Dropout: {DROPOUT}",
        f"LSTM hidden size: {LSTM_HIDDEN_SIZE}",
        f"LSTM layers: {LSTM_LAYERS}",
        f"Bidirectional: {BIDIRECTIONAL}",
        f"Use pooling summary: {USE_POOLING_SUMMARY}",
        "",
        "Overall best run:",
        overall_best.to_string(),
    ]
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\nSaved CNN+LSTM results to:", OUTPUT_DIR)
    print("Best run:")
    print(overall_best.to_string())


if __name__ == "__main__":
    main()