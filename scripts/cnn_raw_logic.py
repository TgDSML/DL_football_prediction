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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.raw_dataset_cnn import CNNDataModule
from src.models.raw_cnn import (
    Conv1DMatchPredictor,
    Conv2DMatchPredictor,
    HybridCNNMatchPredictor,
)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = Path("data/processed/sequences")
OUTPUT_DIR = Path("artifacts/cnn_results/cnn_raw_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANT = "home_away"
SEQUENCE_LENGTHS = [3, 5, 10]
MODEL_TYPES = ["conv1d", "conv2d", "hybrid"]

MAX_EPOCHS = 40
PATIENCE = 8
NUM_CLASSES = 3
NUM_WORKERS = 0

FIXED_BATCH_SIZE = 64
FIXED_LR = 3e-4
FIXED_WEIGHT_DECAY = 1e-3
FIXED_LABEL_SMOOTHING = 0.0
FIXED_DROPOUT = 0.3

FIXED_CONV1D_CHANNELS = [128, 128, 256]
FIXED_CONV2D_CHANNELS = [32, 64, 128]
FIXED_HYBRID_1D_CHANNELS = [128, 256]
FIXED_HYBRID_2D_CHANNELS = [32, 64]

CLASS_NAMES = ["HomeWin", "Draw", "AwayWin"]


@dataclass
class RunResult:
    model_type: str
    sequence_length: int
    run_id: str
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


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_data_module(model_type: str, sequence_length: int) -> CNNDataModule:
    use_cnn_format = model_type == "conv1d"
    return CNNDataModule(
        data_dir=DATA_DIR,
        sequence_length=sequence_length,
        variant=VARIANT,
        batch_size=FIXED_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        normalize=True,
        use_cnn_format=use_cnn_format,
        load_metadata=True,
    )


def build_model(model_type: str, sequence_length: int, num_features: int) -> nn.Module:
    if model_type == "conv1d":
        return Conv1DMatchPredictor(
            input_channels=num_features,
            num_classes=NUM_CLASSES,
            channels=FIXED_CONV1D_CHANNELS,
            dropout=FIXED_DROPOUT,
        )

    if model_type == "conv2d":
        return Conv2DMatchPredictor(
            input_height=sequence_length,
            input_width=num_features,
            input_channels=1,
            num_classes=NUM_CLASSES,
            channels=FIXED_CONV2D_CHANNELS,
            dropout=FIXED_DROPOUT,
        )

    if model_type == "hybrid":
        return HybridCNNMatchPredictor(
            input_channels=num_features,
            sequence_length=sequence_length,
            num_classes=NUM_CLASSES,
            conv1d_channels=FIXED_HYBRID_1D_CHANNELS,
            conv2d_channels=FIXED_HYBRID_2D_CHANNELS,
            dropout=FIXED_DROPOUT,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def to_conv2d_input(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected 3D tensor for conv2d conversion, got {tuple(x.shape)}")
    return x.unsqueeze(1)


def prepare_inputs(x: torch.Tensor, model_type: str) -> torch.Tensor:
    if model_type == "conv2d":
        if x.ndim == 3:
            x = to_conv2d_input(x)
        if x.ndim != 4:
            raise ValueError(f"Conv2D expects (N, 1, k, F), got {tuple(x.shape)}")
    elif model_type == "conv1d":
        if x.ndim != 3:
            raise ValueError(f"Conv1D expects (N, F, k), got {tuple(x.shape)}")
    elif model_type == "hybrid":
        if x.ndim != 3:
            raise ValueError(f"Hybrid expects (N, k, F), got {tuple(x.shape)}")
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return x


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    model_type: str,
) -> dict:
    training = optimizer is not None
    model.train(training)

    losses = []
    all_preds = []
    all_targets = []

    for batch in loader:
        x = batch["sequences"].to(DEVICE)
        y = batch["labels"].to(DEVICE)

        x = prepare_inputs(x, model_type)

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
    model_type: str,
) -> dict | None:
    if loader is None:
        return None
    model.eval()
    with torch.no_grad():
        return run_epoch(model, loader, criterion, optimizer=None, model_type=model_type)


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
        f"Best epoch: {result.best_epoch}",
        "",
        "Primary selection metrics",
        "-------------------------",
        f"Validation Macro F1: {result.val_f1_macro:.6f}",
        f"Validation Accuracy: {result.val_accuracy:.6f}",
        f"Validation Loss: {result.val_loss:.6f}",
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


def fit_single_run(model_type: str, sequence_length: int) -> RunResult:
    run_id = f"{model_type}_seq{sequence_length}"

    data_module = build_data_module(model_type, sequence_length)
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    if val_loader is None:
        raise RuntimeError("Validation loader is required.")

    if data_module.feature_names is None:
        raise RuntimeError("feature_names were not loaded from the NPZ files.")

    num_features = len(data_module.feature_names)

    model = build_model(model_type, sequence_length, num_features).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=FIXED_LABEL_SMOOTHING)
    optimizer = AdamW(model.parameters(), lr=FIXED_LR, weight_decay=FIXED_WEIGHT_DECAY)

    best_state = None
    best_epoch = -1
    best_val_macro_f1 = -1.0
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_val_metrics = None
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, model_type)

        model.eval()
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, optimizer=None, model_type=model_type)

        improved = (
            val_metrics["f1_macro"] > best_val_macro_f1 + 1e-6
            or (
                abs(val_metrics["f1_macro"] - best_val_macro_f1) <= 1e-6
                and val_metrics["accuracy"] > best_val_accuracy + 1e-6
            )
            or (
                abs(val_metrics["f1_macro"] - best_val_macro_f1) <= 1e-6
                and abs(val_metrics["accuracy"] - best_val_accuracy) <= 1e-6
                and val_metrics["loss"] < best_val_loss
            )
        )

        if improved:
            best_val_macro_f1 = val_metrics["f1_macro"]
            best_val_accuracy = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_val_metrics = copy.deepcopy(val_metrics)
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
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
    test_metrics = evaluate_loader(model, test_loader, criterion, model_type=model_type)

    result = RunResult(
        model_type=model_type,
        sequence_length=sequence_length,
        run_id=run_id,
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

    results = []

    for model_type in MODEL_TYPES:
        for sequence_length in SEQUENCE_LENGTHS:
            print(f"\nRunning {model_type} with sequence_length={sequence_length}")
            result = fit_single_run(model_type, sequence_length)
            results.append(asdict(result))
            pd.DataFrame(results).to_csv(OUTPUT_DIR / "results.csv", index=False)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)

    best_by_group = (
        df.sort_values(
            ["model_type", "sequence_length", "val_f1_macro", "val_accuracy", "val_loss"],
            ascending=[True, True, False, False, True],
        )
        .groupby(["model_type", "sequence_length"], as_index=False)
        .first()
    )
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
        "CNN raw results summary",
        "=======================",
        "",
        "Primary ranking metrics:",
        "1. Validation Macro F1",
        "2. Validation Accuracy",
        "3. Validation Loss",
        "",
        "Saved readable outputs:",
        "- results.csv",
        "- best_by_group.csv",
        "- overall_best.json",
        "- *_report.txt",
        "",
        "Best by group:",
        best_by_group.to_string(index=False),
        "",
        "Overall best:",
        overall_best.to_string(),
    ]
    (OUTPUT_DIR / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\nSaved readable results to:", OUTPUT_DIR)
    print("No checkpoints are being saved in this version.")


if __name__ == "__main__":
    main()
