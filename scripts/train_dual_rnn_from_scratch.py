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

from src.features.sequences import HOME_AWAY_PREFIX, SEQUENCE_LENGTH, SEQUENCE_OUTPUT_DIR, TARGET_NAMES
from src.models.rnn_from_scratch import DualScratchRNNClassifier
from src.utils.common import ensure_dir, get_device, set_seed


def parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a dual-encoder scratch RNN baseline.")
    parser.add_argument("--sequence-dir", default=SEQUENCE_OUTPUT_DIR)
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--class-weighting", choices=("none", "balanced"), default="none")
    parser.add_argument("--draw-weight", type=float, default=None)
    parser.add_argument("--shared-encoder", type=parse_bool, default=False)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def prefix_for_sequence_length(sequence_length: int) -> str:
    if sequence_length == SEQUENCE_LENGTH:
        return HOME_AWAY_PREFIX
    return f"{HOME_AWAY_PREFIX}_seq{sequence_length}"


def report_path(experiment_name: str, report_dir: str | None = None) -> Path:
    if report_dir is not None:
        return PROJECT_ROOT / report_dir / f"{experiment_name}.txt"
    return PROJECT_ROOT / f"outputs/reports/dual_rnn_from_scratch_{experiment_name}.txt"


def load_split(
    sequence_dir: Path,
    prefix: str,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    npz_path = sequence_dir / f"{prefix}_{split_name}.npz"
    metadata_path = sequence_dir / f"{prefix}_{split_name}_metadata.csv"
    if not npz_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {prefix} sequence files for {split_name}. Run `python scripts/build_sequences.py --sequence-length <N> --variants home_away` first."
        )

    arrays = np.load(npz_path, allow_pickle=False)
    metadata = pd.read_csv(metadata_path)
    for column in metadata.columns:
        if column.endswith("_date") or column == "Date":
            metadata[column] = pd.to_datetime(metadata[column], errors="raise")
    return arrays["X"], arrays["y"], metadata, arrays["feature_names"].tolist()


def validate_and_split_home_away(
    X: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    if X.ndim != 3:
        raise ValueError(f"Expected X shape [samples, sequence_length, features], got {X.shape}")
    if X.shape[2] != len(feature_names):
        raise ValueError(
            f"Feature-name count {len(feature_names)} does not match X feature dimension {X.shape[2]}"
        )
    if X.shape[2] % 2 != 0:
        raise ValueError(f"Home-away feature dimension must be even, got {X.shape[2]}")

    input_dim = X.shape[2] // 2
    home_features = feature_names[:input_dim]
    away_features = feature_names[input_dim:]
    if not home_features or not away_features:
        raise ValueError("Home-away split produced an empty feature set.")
    if not all(name.startswith("home_") for name in home_features):
        raise ValueError("Expected first half of feature names to start with 'home_'.")
    if not all(name.startswith("away_") for name in away_features):
        raise ValueError("Expected second half of feature names to start with 'away_'.")

    stripped_home = [name.removeprefix("home_") for name in home_features]
    stripped_away = [name.removeprefix("away_") for name in away_features]
    if stripped_home != stripped_away:
        raise ValueError("Home and away feature names do not align after prefix removal.")

    return X[:, :, :input_dim], X[:, :, input_dim:], home_features, away_features


def standardize_streams(
    train_home: np.ndarray,
    val_home: np.ndarray,
    test_home: np.ndarray,
    train_away: np.ndarray,
    val_away: np.ndarray,
    test_away: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    home_mean = train_home.mean(axis=(0, 1), keepdims=True)
    home_std = np.where(train_home.std(axis=(0, 1), keepdims=True) < 1e-6, 1.0, train_home.std(axis=(0, 1), keepdims=True))
    away_mean = train_away.mean(axis=(0, 1), keepdims=True)
    away_std = np.where(train_away.std(axis=(0, 1), keepdims=True) < 1e-6, 1.0, train_away.std(axis=(0, 1), keepdims=True))

    return (
        ((train_home - home_mean) / home_std).astype(np.float32, copy=False),
        ((val_home - home_mean) / home_std).astype(np.float32, copy=False),
        ((test_home - home_mean) / home_std).astype(np.float32, copy=False),
        ((train_away - away_mean) / away_std).astype(np.float32, copy=False),
        ((val_away - away_mean) / away_std).astype(np.float32, copy=False),
        ((test_away - away_mean) / away_std).astype(np.float32, copy=False),
    )


def make_loader(
    home_X: np.ndarray,
    away_X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(home_X, dtype=torch.float32),
        torch.tensor(away_X, dtype=torch.float32),
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
    weights = float(counts.sum()) / (len(counts) * counts)
    if draw_weight is not None:
        weights[1] *= float(draw_weight)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(
    model: DualScratchRNNClassifier,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    losses: list[float] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []

    with torch.no_grad():
        for home_batch, away_batch, y_batch in data_loader:
            home_batch = home_batch.to(device)
            away_batch = away_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(home_batch, away_batch)
            losses.append(float(criterion(logits, y_batch).item()))
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
    model: DualScratchRNNClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    lr: float,
    gradient_clip: float,
) -> tuple[DualScratchRNNClassifier, list[dict[str, float]]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_macro_f1 = float("-inf")

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for home_batch, away_batch, y_batch in train_loader:
            home_batch = home_batch.to(device)
            away_batch = away_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(home_batch, away_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            if gradient_clip > 0:
                clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        val_metrics = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(batch_losses)) if batch_losses else 0.0,
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
    args: argparse.Namespace,
    prefix: str,
    feature_names: list[str],
    train_meta: pd.DataFrame,
    val_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    train_home: np.ndarray,
    train_away: np.ndarray,
    history: list[dict[str, float]],
    train_metrics: dict[str, object],
    val_metrics: dict[str, object],
    test_metrics: dict[str, object],
    counts: np.ndarray,
    weights: torch.Tensor | None,
) -> str:
    weights_text = "none" if weights is None else np.array2string(weights.detach().cpu().numpy(), precision=6)
    accuracy_gap = float(train_metrics["accuracy"]) - float(val_metrics["accuracy"])
    macro_f1_gap = float(train_metrics["macro_f1"]) - float(val_metrics["macro_f1"])
    lines = [
        "Dual From-Scratch RNN Report",
        "=" * 32,
        f"Experiment name: {args.experiment_name}",
        f"Sequence prefix: {prefix}",
        f"Sequence length: {args.sequence_length}",
        f"Hidden dim: {args.hidden_dim}",
        f"Dropout: {args.dropout}",
        f"Shared encoder: {args.shared_encoder}",
        f"Class weighting: {args.class_weighting}",
        f"Draw weight multiplier: {args.draw_weight if args.draw_weight is not None else 'none'}",
        f"Class counts [HomeWin, Draw, AwayWin]: {counts.tolist()}",
        f"Class weights [HomeWin, Draw, AwayWin]: {weights_text}",
        f"Home feature names: {', '.join(feature_names[:train_home.shape[2]])}",
        f"Away feature names: {', '.join(feature_names[train_home.shape[2]:])}",
        f"Train home X shape: {train_home.shape}",
        f"Train away X shape: {train_away.shape}",
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
            "Train Metrics",
            "=" * 32,
            f"Accuracy: {float(train_metrics['accuracy']):.4f}",
            f"Macro F1: {float(train_metrics['macro_f1']):.4f}",
            f"Accuracy Gap Train-Val: {accuracy_gap:.4f}",
            f"Macro F1 Gap Train-Val: {macro_f1_gap:.4f}",
            "Confusion Matrix:",
            np.array2string(np.asarray(train_metrics["confusion_matrix"])),
            "Classification Report:",
            str(train_metrics["classification_report"]).rstrip(),
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
    prefix = prefix_for_sequence_length(args.sequence_length)

    train_X, train_y, train_meta, feature_names = load_split(sequence_dir, prefix, "train")
    val_X, val_y, val_meta, val_feature_names = load_split(sequence_dir, prefix, "val")
    test_X, test_y, test_meta, test_feature_names = load_split(sequence_dir, prefix, "test")
    if feature_names != val_feature_names or feature_names != test_feature_names:
        raise ValueError("Feature name mismatch across saved sequence splits.")

    train_home, train_away, _, _ = validate_and_split_home_away(train_X, feature_names)
    val_home, val_away, _, _ = validate_and_split_home_away(val_X, feature_names)
    test_home, test_away, _, _ = validate_and_split_home_away(test_X, feature_names)
    train_home, val_home, test_home, train_away, val_away, test_away = standardize_streams(
        train_home,
        val_home,
        test_home,
        train_away,
        val_away,
        test_away,
    )

    train_loader = make_loader(train_home, train_away, train_y, args.batch_size, shuffle=True)
    train_eval_loader = make_loader(train_home, train_away, train_y, args.batch_size, shuffle=False)
    val_loader = make_loader(val_home, val_away, val_y, args.batch_size, shuffle=False)
    test_loader = make_loader(test_home, test_away, test_y, args.batch_size, shuffle=False)

    counts = class_counts(train_y)
    weights = make_class_weights(train_y, args.class_weighting, args.draw_weight)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None)

    model = DualScratchRNNClassifier(
        input_dim=train_home.shape[2],
        hidden_dim=args.hidden_dim,
        num_classes=3,
        dropout=args.dropout,
        shared_encoder=args.shared_encoder,
    ).to(device)
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        gradient_clip=args.gradient_clip,
    )
    train_metrics = evaluate(model, train_eval_loader, criterion, device)
    val_metrics = evaluate(model, val_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)

    output_path = report_path(args.experiment_name, args.report_dir)
    ensure_dir(output_path.parent)
    output_path.write_text(
        build_report(
            args=args,
            prefix=prefix,
            feature_names=feature_names,
            train_meta=train_meta,
            val_meta=val_meta,
            test_meta=test_meta,
            train_home=train_home,
            train_away=train_away,
            history=history,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            counts=counts,
            weights=weights,
        ),
        encoding="utf-8",
    )

    print(f"Device: {device}")
    print(f"Train home X shape: {train_home.shape}, away X shape: {train_away.shape}")
    print(f"Train accuracy: {float(train_metrics['accuracy']):.4f}")
    print(f"Train macro F1: {float(train_metrics['macro_f1']):.4f}")
    print(f"Val accuracy: {float(val_metrics['accuracy']):.4f}")
    print(f"Val macro F1: {float(val_metrics['macro_f1']):.4f}")
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
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
