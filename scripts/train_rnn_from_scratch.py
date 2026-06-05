from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.sequences import HOME_ONLY_PREFIX, SEQUENCE_OUTPUT_DIR, TARGET_NAMES
from src.models.rnn_from_scratch import ScratchRNNClassifier
from src.utils.common import ensure_dir, get_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the home-only from-scratch vanilla RNN baseline."
    )
    parser.add_argument("--sequence-dir", default=SEQUENCE_OUTPUT_DIR)
    parser.add_argument("--prefix", default=HOME_ONLY_PREFIX)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--class-weighting", choices=("none", "balanced"), default="none")
    parser.add_argument("--draw-weight", type=float, default=None)
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-path", default=None)
    return parser.parse_args()


def default_report_path(prefix: str) -> Path:
    return PROJECT_ROOT / f"outputs/reports/rnn_from_scratch_{prefix}_report.txt"


def experiment_report_path(experiment_name: str) -> Path:
    return PROJECT_ROOT / f"outputs/reports/rnn_from_scratch_{experiment_name}_report.txt"


def load_split(sequence_dir: Path, prefix: str, split_name: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    npz_path = sequence_dir / f"{prefix}_{split_name}.npz"
    metadata_path = sequence_dir / f"{prefix}_{split_name}_metadata.csv"
    if not npz_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing sequence files for {split_name}. Run `python scripts/build_sequences.py` first."
        )

    arrays = np.load(npz_path, allow_pickle=False)
    metadata = pd.read_csv(metadata_path)
    for column in metadata.columns:
        if column.endswith("_date") or column == "Date":
            metadata[column] = pd.to_datetime(metadata[column], errors="raise")
    feature_names = arrays["feature_names"].tolist()
    return arrays["X"], arrays["y"], metadata, feature_names


def standardize_sequences(
    train_X: np.ndarray,
    val_X: np.ndarray,
    test_X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_X.mean(axis=(0, 1), keepdims=True)
    std = train_X.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    train_scaled = ((train_X - mean) / std).astype(np.float32, copy=False)
    val_scaled = ((val_X - mean) / std).astype(np.float32, copy=False)
    test_scaled = ((test_X - mean) / std).astype(np.float32, copy=False)
    return train_scaled, val_scaled, test_scaled, mean.squeeze(), std.squeeze()


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def class_counts(y_train: np.ndarray) -> np.ndarray:
    return np.bincount(y_train.astype(np.int64), minlength=3)


def make_class_weights(
    y_train: np.ndarray,
    class_weighting: str,
    draw_weight: float | None,
) -> torch.Tensor | None:
    if class_weighting == "none":
        return None

    counts = class_counts(y_train).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError(f"Cannot compute balanced class weights with zero class counts: {counts}")

    total = float(counts.sum())
    weights = total / (len(counts) * counts)
    if draw_weight is not None:
        weights[1] *= float(draw_weight)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(
    model: ScratchRNNClassifier,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            losses.append(float(loss.item()))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(y_batch.cpu().numpy())

    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=TARGET_NAMES,
            zero_division=0,
        ),
    }


def train_model(
    model: ScratchRNNClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    gradient_clip: float,
    criterion: nn.Module,
) -> tuple[ScratchRNNClassifier, list[dict[str, float]]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_macro_f1 = float("-inf")

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses: list[float] = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            if gradient_clip > 0:
                clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        val_metrics = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": float(val_metrics["loss"]),
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
            }
        )

        if float(val_metrics["macro_f1"]) > best_val_macro_f1:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def build_report(
    experiment_name: str,
    feature_names: list[str],
    train_meta: pd.DataFrame,
    val_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    train_X: np.ndarray,
    val_X: np.ndarray,
    test_X: np.ndarray,
    history: list[dict[str, float]],
    val_metrics: dict[str, object],
    test_metrics: dict[str, object],
    counts: np.ndarray,
    weights: torch.Tensor | None,
    class_weighting: str,
    draw_weight: float | None,
) -> str:
    weights_text = "none" if weights is None else np.array2string(weights.detach().cpu().numpy(), precision=6)
    lines = [
        "From-Scratch Vanilla RNN Report",
        "=" * 32,
        f"Experiment name: {experiment_name}",
        f"Class weighting: {class_weighting}",
        f"Draw weight multiplier: {draw_weight if draw_weight is not None else 'none'}",
        f"Class counts [HomeWin, Draw, AwayWin]: {counts.tolist()}",
        f"Class weights [HomeWin, Draw, AwayWin]: {weights_text}",
        f"Sequence feature names: {', '.join(feature_names)}",
        f"Train X shape: {train_X.shape}",
        f"Val X shape: {val_X.shape}",
        f"Test X shape: {test_X.shape}",
        "",
        "Date Ranges",
        "=" * 32,
        f"Train: {train_meta['Date'].min().date()} to {train_meta['Date'].max().date()}",
        f"Val: {val_meta['Date'].min().date()} to {val_meta['Date'].max().date()}",
        f"Test: {test_meta['Date'].min().date()} to {test_meta['Date'].max().date()}",
        "",
        "Training History",
        "=" * 32,
    ]

    for epoch_metrics in history:
        lines.append(
            "Epoch "
            f"{int(epoch_metrics['epoch']):02d}: "
            f"train_loss={epoch_metrics['train_loss']:.4f} "
            f"val_loss={epoch_metrics['val_loss']:.4f} "
            f"val_acc={epoch_metrics['val_accuracy']:.4f} "
            f"val_macro_f1={epoch_metrics['val_macro_f1']:.4f}"
        )

    lines.extend(
        [
            "",
            "Validation Metrics",
            "=" * 32,
            f"Accuracy: {float(val_metrics['accuracy']):.4f}",
            f"Macro F1: {float(val_metrics['macro_f1']):.4f}",
            "Confusion Matrix:",
            np.array2string(np.asarray(val_metrics["confusion_matrix"])),
            "Classification Report:",
            str(val_metrics["classification_report"]).rstrip(),
            "",
            "Test Metrics",
            "=" * 32,
            f"Accuracy: {float(test_metrics['accuracy']):.4f}",
            f"Macro F1: {float(test_metrics['macro_f1']):.4f}",
            "Confusion Matrix:",
            np.array2string(np.asarray(test_metrics["confusion_matrix"])),
            "Classification Report:",
            str(test_metrics["classification_report"]).rstrip(),
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    sequence_dir = Path(args.sequence_dir)
    experiment_name = args.experiment_name or args.prefix
    if args.report_path:
        report_path = Path(args.report_path)
    elif args.experiment_name:
        report_path = experiment_report_path(args.experiment_name)
    else:
        report_path = default_report_path(args.prefix)

    train_X, train_y, train_meta, feature_names = load_split(sequence_dir, args.prefix, "train")
    val_X, val_y, val_meta, val_feature_names = load_split(sequence_dir, args.prefix, "val")
    test_X, test_y, test_meta, test_feature_names = load_split(sequence_dir, args.prefix, "test")

    if feature_names != val_feature_names or feature_names != test_feature_names:
        raise ValueError("Feature name mismatch across saved sequence splits.")

    train_X, val_X, test_X, _, _ = standardize_sequences(train_X, val_X, test_X)

    train_loader = make_loader(train_X, train_y, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(val_X, val_y, batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(test_X, test_y, batch_size=args.batch_size, shuffle=False)

    model = ScratchRNNClassifier(
        input_dim=train_X.shape[2],
        hidden_dim=args.hidden_dim,
        num_classes=3,
    ).to(device)
    counts = class_counts(train_y)
    weights = make_class_weights(train_y, args.class_weighting, args.draw_weight)
    criterion = nn.CrossEntropyLoss(
        weight=weights.to(device) if weights is not None else None
    )

    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        gradient_clip=args.gradient_clip,
        criterion=criterion,
    )

    val_metrics = evaluate(model, val_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)

    report = build_report(
        experiment_name=experiment_name,
        feature_names=feature_names,
        train_meta=train_meta,
        val_meta=val_meta,
        test_meta=test_meta,
        train_X=train_X,
        val_X=val_X,
        test_X=test_X,
        history=history,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        counts=counts,
        weights=weights,
        class_weighting=args.class_weighting,
        draw_weight=args.draw_weight,
    )

    ensure_dir(report_path.parent)
    report_path.write_text(report, encoding="utf-8")

    print(f"Device: {device}")
    print(f"Train X shape: {train_X.shape}, y shape: {train_y.shape}")
    print(f"Val X shape: {val_X.shape}, y shape: {val_y.shape}")
    print(f"Test X shape: {test_X.shape}, y shape: {test_y.shape}")
    print(f"Feature names: {', '.join(feature_names)}")
    print(f"Validation accuracy: {float(val_metrics['accuracy']):.4f}")
    print(f"Validation macro F1: {float(val_metrics['macro_f1']):.4f}")
    print(f"Test accuracy: {float(test_metrics['accuracy']):.4f}")
    print(f"Test macro F1: {float(test_metrics['macro_f1']):.4f}")
    print(f"Class counts [HomeWin, Draw, AwayWin]: {counts.tolist()}")
    print(
        "Class weights [HomeWin, Draw, AwayWin]: "
        f"{'none' if weights is None else weights.detach().cpu().numpy().tolist()}"
    )
    print("Test confusion matrix:")
    print(np.array2string(np.asarray(test_metrics["confusion_matrix"])))
    print("Test classification report:")
    print(str(test_metrics["classification_report"]).rstrip())
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
