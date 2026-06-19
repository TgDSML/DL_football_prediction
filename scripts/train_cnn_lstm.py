from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import copy
import json
import random
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn_lstm import CNNLSTMMatchPredictor
from src.raw_dataset_cnn import CNNDataModule

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = Path("data/processed/sequences_leakage_safe")
OUTPUT_DIR = Path("artifacts/cnn_lstm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANT = "home_away"
SEQUENCE_LENGTHS = [3, 5, 10]
NUM_CLASSES = 3
CLASS_NAMES = ["HomeWin", "Draw", "AwayWin"]

BATCH_SIZE = 64
LEARNING_RATE = 3e-4
DROPOUT = 0.3
WEIGHT_DECAY = 1e-3
MAX_EPOCHS = 40
PATIENCE = 8
GRAD_CLIP_NORM = 5.0
NUM_WORKERS = 0


@dataclass
class RunResult:
    run_id: str
    sequence_length: int
    learning_rate: float
    dropout: float
    batch_size: int
    best_epoch: int
    val_loss: float
    val_accuracy: float
    val_f1_macro: float
    val_f1_weighted: float
    test_loss: float
    test_accuracy: float
    test_f1_macro: float
    test_f1_weighted: float
    report_path: str


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


def build_model(num_features: int, dropout: float) -> nn.Module:
    return CNNLSTMMatchPredictor(
        input_features=num_features,
        num_classes=NUM_CLASSES,
        conv_channels=[64, 128],
        kernel_size=3,
        lstm_hidden_size=128,
        lstm_layers=1,
        dropout=dropout,
        bidirectional=False,
    )


def prepare_inputs(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"Expected raw sequence tensor with shape (N, k, F), got {tuple(x.shape)}")
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
        x = prepare_inputs(batch["sequences"].to(DEVICE))
        y = batch["labels"].to(DEVICE)

        logits = model(x)
        loss = criterion(logits, y)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

        preds = logits.argmax(dim=1)
        losses.append(float(loss.detach().cpu().item()))
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
    val_metrics: dict,
    test_metrics: dict | None,
) -> Path:
    report_path = OUTPUT_DIR / f"{result.run_id}_report.txt"
    lines = [
        f"Run ID: {result.run_id}",
        f"Sequence length: {result.sequence_length}",
        f"Learning rate: {result.learning_rate}",
        f"Dropout: {result.dropout}",
        f"Batch size: {result.batch_size}",
        f"Best epoch: {result.best_epoch}",
        "",
        "Selection rule:",
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
    run_id = f"cnn_lstm_seq{sequence_length}"

    data_module = build_data_module(sequence_length, BATCH_SIZE)
    data_module.setup()

    if data_module.val_dataset is None:
        raise RuntimeError("Validation dataset is required for model selection.")
    if data_module.test_dataset is None:
        raise RuntimeError("Test dataset is required for final evaluation.")

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    num_features = (
        len(data_module.feature_names)
        if data_module.feature_names is not None
        else int(data_module.train_dataset.sequences.shape[2])
    )

    model = build_model(num_features, DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = -1
    best_val_f1 = -1.0
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_val_metrics: dict | None = None
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer)

        model.eval()
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, criterion, optimizer=None)

        if is_better(val_metrics, best_val_f1, best_val_acc, best_val_loss):
            best_val_f1 = val_metrics["f1_macro"]
            best_val_acc = val_metrics["accuracy"]
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

    if test_metrics is None:
        raise RuntimeError("Test evaluation failed.")

    result = RunResult(
        run_id=run_id,
        sequence_length=sequence_length,
        learning_rate=LEARNING_RATE,
        dropout=DROPOUT,
        batch_size=BATCH_SIZE,
        best_epoch=best_epoch,
        val_loss=best_val_loss,
        val_accuracy=best_val_acc,
        val_f1_macro=best_val_f1,
        val_f1_weighted=best_val_metrics["f1_weighted"],
        test_loss=test_metrics["loss"],
        test_accuracy=test_metrics["accuracy"],
        test_f1_macro=test_metrics["f1_macro"],
        test_f1_weighted=test_metrics["f1_weighted"],
        report_path="",
    )

    report_path = write_run_report(result, best_val_metrics, test_metrics)
    result.report_path = str(report_path)
    return result


def save_results(results: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "results.csv", index=False)
    return df


def save_best_by_group(full_df: pd.DataFrame) -> pd.DataFrame:
    ranked = full_df.sort_values(
        ["val_f1_macro", "val_accuracy", "val_loss"],
        ascending=[False, False, True],
    )
    best_by_group = ranked.groupby("sequence_length", as_index=False).first()
    best_by_group.to_csv(OUTPUT_DIR / "best_by_group.csv", index=False)
    return best_by_group


def save_overall_best(full_df: pd.DataFrame) -> dict:
    ranked = full_df.sort_values(
        ["val_f1_macro", "val_accuracy", "val_loss"],
        ascending=[False, False, True],
    )
    overall_best = ranked.iloc[0].to_dict()
    overall_best_json = {
        k: (v.item() if isinstance(v, (np.integer, np.floating)) else v)
        for k, v in overall_best.items()
    }
    (OUTPUT_DIR / "overall_best.json").write_text(
        json.dumps(overall_best_json, indent=2), encoding="utf-8"
    )
    return overall_best_json


def write_summary(
    results_df: pd.DataFrame,
    best_by_group_df: pd.DataFrame,
    overall_best: dict,
) -> None:
    ranking_rule = [
        "1. Validation Macro F1",
        "2. Validation Accuracy",
        "3. Validation Loss",
    ]

    lines = [
        "CNN+LSTM training summary",
        "========================",
        "",
        "Selection rule:",
        *ranking_rule,
        "",
        f"Best sequence length: {overall_best['sequence_length']}",
        f"Best run ID: {overall_best['run_id']}",
        "",
        "Best validation metrics:",
        f"  Loss: {overall_best['val_loss']:.6f}",
        f"  Accuracy: {overall_best['val_accuracy']:.6f}",
        f"  Macro F1: {overall_best['val_f1_macro']:.6f}",
        f"  Weighted F1: {overall_best['val_f1_weighted']:.6f}",
        "",
        "Best test metrics:",
        f"  Loss: {overall_best['test_loss']:.6f}",
        f"  Accuracy: {overall_best['test_accuracy']:.6f}",
        f"  Macro F1: {overall_best['test_f1_macro']:.6f}",
        f"  Weighted F1: {overall_best['test_f1_weighted']:.6f}",
        "",
        "Per-sequence-length best runs:",
        best_by_group_df.to_string(index=False),
        "",
        "Saved files:",
        "- results.csv",
        "- best_by_group.csv",
        "- overall_best.json",
        "- summary.txt",
        "- *_report.txt",
    ]

    (OUTPUT_DIR / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_seed(SEED)

    run_results: list[dict[str, object]] = []
    for sequence_length in SEQUENCE_LENGTHS:
        print(f"\nRunning CNN+LSTM for sequence length {sequence_length}")
        result = fit_single_run(sequence_length)
        run_results.append(asdict(result))

    results_df = save_results(run_results)
    best_by_group_df = save_best_by_group(results_df)
    overall_best = save_overall_best(results_df)
    write_summary(results_df, best_by_group_df, overall_best)

    print(f"\nSaved outputs to {OUTPUT_DIR}")
    print("Completed training and evaluation for CNN+LSTM.")


if __name__ == "__main__":
    main()

