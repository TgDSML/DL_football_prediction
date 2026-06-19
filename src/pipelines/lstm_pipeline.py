"""End-to-end LSTM pipeline for football match prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.config import ExperimentConfig
from src.dataset import FootballSequenceDataset, MatchBatch
from src.evaluation.metrics import compute_metrics
from src.features.sequences import SEQUENCE_FEATURES, build_team_form_sequences
from src.model import build_model
from src.utils import ensure_dir, get_device, set_seed


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.

    Focuses learning on hard examples by down-weighting easy ones.
    L = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self, alpha: torch.Tensor | None = None, gamma: float = 2.0, reduction: str = "mean"
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Logits of shape (batch_size, num_classes).
            targets: Labels of shape (batch_size,).

        Returns:
            Focal loss value.
        """
        p = torch.softmax(inputs, dim=1)
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction="none", weight=self.alpha
        )
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class TrainingMetrics(NamedTuple):
    """Metrics for a training epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_f1: float
    val_macro_f1: float


class LSTMPipeline:
    """End-to-end LSTM training and evaluation pipeline."""

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize pipeline with configuration.

        Args:
            config: Experiment configuration containing paths, model parameters, and training settings.
        """
        self.config = config
        self.device = get_device()
        self.metrics_history: list[TrainingMetrics] = []
        self.model: nn.Module | None = None
        self.best_val_loss = float("inf")
        self.label_encoder: LabelEncoder | None = None
        self.class_names: list[str] = []

        # Setup directories
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create output directories."""
        output_dirs = self.config.get("outputs", default={})
        for directory in output_dirs.values():
            ensure_dir(str(directory))

    @staticmethod
    def _collate_fn(batch: list[MatchBatch]) -> MatchBatch:
        """Custom collate function for MatchBatch objects.

        Args:
            batch: List of MatchBatch objects.

        Returns:
            Batched MatchBatch with stacked tensors.
        """
        sequences = torch.stack([item.sequences for item in batch])
        labels = torch.stack([item.labels for item in batch])
        return MatchBatch(sequences=sequences, labels=labels)

    def load_and_prepare_data(
        self, raw_data_path: str | Path
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load raw match data and create sequences.

        Args:
            raw_data_path: Path to raw match data CSV.

        Returns:
            Tuple of (sequences_array, labels_array, feature_columns, match_dates).
        """
        print(f"Loading raw data from {raw_data_path}...")
        raw_path = Path(raw_data_path)

        # Support passing a directory, glob pattern, or single file.
        if raw_path.is_dir():
            csv_files = sorted(raw_path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files found in directory: {raw_path}")
            df = pd.concat((pd.read_csv(p) for p in csv_files), ignore_index=True)
        elif "*" in str(raw_data_path):
            csv_files = sorted(Path().glob(str(raw_data_path)))
            if not csv_files:
                raise FileNotFoundError(f"No CSV files match pattern: {raw_data_path}")
            df = pd.concat((pd.read_csv(p) for p in csv_files), ignore_index=True)
        else:
            if not raw_path.exists():
                raise FileNotFoundError(f"Raw data file not found: {raw_data_path}")
            df = pd.read_csv(raw_path)

        # Extract features and target
        feature_cols = self.config.get("data", "feature_columns", default=None)
        target_col = self.config.get("data", "target_column", default="FTR")
        sequence_length = int(self.config.get("data", "sequence_length", default=5))

        model_type = str(self.config.get("training", "model_type", default="lstm"))
        if feature_cols is None and model_type == "lstm":
            # Use canonical strict-prior team-history features. The sequence
            # builder excludes the target fixture row before selecting features.
            feature_cols = list(SEQUENCE_FEATURES)
        elif feature_cols is None:
            # Auto-detect numeric columns (excluding known non-feature columns)
            exclude_cols = {"Date", "HomeTeam", "AwayTeam", "FTR", "HTR", "result"}
            feature_cols = [
                col for col in df.columns if df[col].dtype in ["float64", "int64"]
            ]
            feature_cols = [c for c in feature_cols if c not in exclude_cols]
        else:
            # If custom features are provided for LSTM, ensure goal/result columns are removed.
            if model_type == "lstm":
                feature_cols = [
                    c for c in feature_cols if c not in {"FTHG", "FTAG", "HTHG", "HTAG"}
                ]

        print(f"Using features: {feature_cols}")
        print(f"Sequence length: {sequence_length}")
        print("Note: Features will be standardized using training data only to avoid leakage.")

        # Build sequences for home and away teams, aligned by actual match index
        home_sequences = build_team_form_sequences(
            df, "HomeTeam", "Date", feature_cols, sequence_length
        )
        away_sequences = build_team_form_sequences(
            df, "AwayTeam", "Date", feature_cols, sequence_length
        )

        # Pair home/away history for matches where both teams have enough prior form
        valid_match_indices = sorted(
            set(home_sequences.keys()).intersection(away_sequences.keys())
        )
        sequences_list = []
        labels_list = []
        dates_list: list[object] = []
        for match_idx in valid_match_indices:
            home_seq = home_sequences[match_idx].values
            away_seq = away_sequences[match_idx].values
            combined_seq = np.concatenate([home_seq, away_seq], axis=1)
            sequences_list.append(combined_seq)
            labels_list.append(df.loc[match_idx, target_col])
            if "Date" in df.columns:
                dates_list.append(df.loc[match_idx, "Date"])
            else:
                dates_list.append(match_idx)

        sequences_array = np.array(sequences_list)
        labels_array = np.array(labels_list)

        if "Date" in df.columns:
            match_dates = pd.to_datetime(pd.Series(dates_list), dayfirst=True, errors="coerce").values
        else:
            match_dates = np.array(dates_list, dtype=int)

        # Encode labels if they are strings
        if labels_array.dtype == object or isinstance(labels_array[0], str):
            self.label_encoder = LabelEncoder()
            self.class_names = sorted(list(set(labels_array)))
            labels_array_encoded = self.label_encoder.fit_transform(labels_array)
            print(f"Encoded labels: {dict(zip(self.class_names, self.label_encoder.transform(self.class_names)))}")
        else:
            labels_array_encoded = labels_array.astype(int)
            self.class_names = [str(i) for i in sorted(set(labels_array))]

        print(f"Created {len(sequences_array)} sequences of shape {sequences_array[0].shape}")
        print(f"Labels distribution: {np.unique(labels_array_encoded, return_counts=True)}")

        self.input_size = sequences_array.shape[-1]
        return sequences_array, labels_array_encoded, feature_cols, match_dates

    def create_dataloaders(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        match_dates: np.ndarray | None = None,
    ) -> tuple[DataLoader[MatchBatch], DataLoader[MatchBatch], DataLoader[MatchBatch]]:
        """Create train/val/test dataloaders using a time-based split.

        Args:
            sequences: Sequence array of shape (num_samples, seq_len, num_features).
            labels: Label array of shape (num_samples,).
            match_dates: Array providing match ordering for each sample.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        batch_size = int(self.config.get("training", "batch_size", default=32))
        val_size = float(self.config.get("training", "validation_size", default=0.2))
        test_size = float(self.config.get("training", "test_size", default=0.1))

        labels_arr = np.array(labels)

        if match_dates is None:
            match_dates = np.arange(len(sequences))

        # Sort samples by match date before splitting to prevent temporal leakage.
        order = np.argsort(match_dates)
        sequences = sequences[order]
        labels_arr = labels_arr[order]

        num_samples = len(sequences)
        test_count = int(np.floor(test_size * num_samples))
        val_count = int(np.floor(val_size * num_samples))
        train_count = num_samples - val_count - test_count

        if train_count <= 0:
            raise ValueError(
                "Not enough samples for time-based split. "
                "Reduce validation/test sizes or use more data."
            )

        train_idx = np.arange(0, train_count)
        val_idx = np.arange(train_count, train_count + val_count)
        test_idx = np.arange(train_count + val_count, num_samples)

        # Standardize using training data only to avoid leaking test/validation statistics.
        scaler = StandardScaler()
        num_features = sequences.shape[-1]
        train_flat = sequences[train_idx].reshape(-1, num_features)
        scaler.fit(train_flat)
        sequences = scaler.transform(sequences.reshape(-1, num_features)).reshape(sequences.shape)

        from torch.utils.data import Subset

        dataset = FootballSequenceDataset(sequences, labels_arr)
        train_set = Subset(dataset, train_idx)
        val_set = Subset(dataset, val_idx)
        test_set = Subset(dataset, test_idx)

        # Compute class weights using training labels only.
        num_classes = int(self.config.get("model", "num_classes", default=3))
        train_counts = np.bincount(labels_arr[train_idx], minlength=num_classes)
        train_total = train_counts.sum()
        train_counts = train_counts.astype(float) + 1e-6
        class_weights = train_total / (train_counts * num_classes)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

        sample_weights = self.class_weights[labels_arr[train_idx]].numpy()
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_idx), replacement=True
        )

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self._collate_fn,
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False, collate_fn=self._collate_fn
        )
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, collate_fn=self._collate_fn
        )

        print(
            f"Created dataloaders: "
            f"train={len(train_set)}, val={len(val_set)}, test={len(test_set)}"
        )

        return train_loader, val_loader, test_loader

    def build_model(self) -> nn.Module:
        """Build LSTM model from config.

        Returns:
            Initialized LSTM model on device.
        """
        model_type = str(self.config.get("training", "model_type", default="lstm"))
        model_kwargs = dict(self.config.get("model", default={}))

        if "input_size" not in model_kwargs:
            if not hasattr(self, "input_size") or self.input_size is None:
                raise ValueError(
                    "input_size must be defined either in config or inferred from the data"
                )
            model_kwargs["input_size"] = self.input_size
        else:
            if hasattr(self, "input_size") and self.input_size is not None:
                if int(model_kwargs["input_size"]) != int(self.input_size):
                    print(
                        f"Warning: config input_size={model_kwargs['input_size']} "
                        f"does not match inferred input size={self.input_size}. "
                        "Using inferred value."
                    )
                    model_kwargs["input_size"] = self.input_size

        print(f"Building {model_type} model with params: {model_kwargs}")
        model = build_model(model_type, **model_kwargs).to(self.device)
        self.model = model
        return model

    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader[MatchBatch],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Train for one epoch.

        Args:
            model: Model to train.
            train_loader: Training dataloader.
            optimizer: Optimizer.
            criterion: Loss criterion.

        Returns:
            Average training loss.
        """
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            sequences = batch.sequences.to(self.device)
            labels = batch.labels.to(self.device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader[MatchBatch],
        criterion: nn.Module,
    ) -> tuple[float, dict]:
        """Evaluate model on loader.

        Args:
            model: Model to evaluate.
            loader: Data loader.
            criterion: Loss criterion.

        Returns:
            Tuple of (average_loss, metrics_dict).
        """
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                sequences = batch.sequences.to(self.device)
                labels = batch.labels.to(self.device)

                outputs = model(sequences)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        metrics = compute_metrics(np.array(all_labels), np.array(all_preds))

        return avg_loss, metrics

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader[MatchBatch],
        val_loader: DataLoader[MatchBatch],
    ) -> None:
        """Train model with early stopping.

        Args:
            model: Model to train.
            train_loader: Training dataloader.
            val_loader: Validation dataloader.
        """
        epochs = int(self.config.get("training", "epochs", default=10))
        learning_rate = float(self.config.get("training", "learning_rate", default=0.001))
        patience = int(self.config.get("training", "patience", default=5))
        early_stopping = bool(self.config.get("training", "early_stopping", default=True))

        optimizer = Adam(model.parameters(), lr=learning_rate)
        # Use focal loss for better handling of hard examples
        criterion = FocalLoss(alpha=self.class_weights.to(self.device), gamma=2.0)

        # Scheduler to reduce LR on plateau of validation F1
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=max(1, patience // 2))

        patience_counter = 0
        # Monitor validation F1 (higher is better)
        best_metric = float("-inf")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_metrics = self.evaluate(model, val_loader, criterion)

            val_f1 = val_metrics.get("f1", 0.0)
            val_macro_f1 = val_metrics.get("macro_f1", 0.0)
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_metrics.get("accuracy", 0.0),
                val_f1=val_f1,
                val_macro_f1=val_macro_f1,
            )
            self.metrics_history.append(metrics)

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_metrics.get('accuracy', 0.0):.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"Val Macro F1: {val_macro_f1:.4f}"
            )

            # Scheduler step (monitor validation F1)
            try:
                scheduler.step(val_f1)
            except Exception:
                pass

            # Early stopping on validation F1
            if val_f1 > best_metric:
                best_metric = val_f1
                patience_counter = 0
                self._save_checkpoint(model)
            else:
                patience_counter += 1
                if early_stopping and patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    def _save_checkpoint(self, model: nn.Module) -> None:
        """Save model checkpoint."""
        model_dir = Path(self.config.get("outputs", "model_dir", default="outputs/models"))
        ensure_dir(str(model_dir))
        checkpoint_path = model_dir / "best_lstm_model.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def predict(
        self,
        model: nn.Module,
        loader: DataLoader[MatchBatch],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions on loader.

        Args:
            model: Trained model.
            loader: Data loader.

        Returns:
            Tuple of (predictions, probabilities, true_labels).
        """
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                sequences = batch.sequences.to(self.device)
                labels = batch.labels.to(self.device)

                outputs = model(sequences)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return (
            np.array(all_preds),
            np.array(all_probs),
            np.array(all_labels),
        )

    def save_results(
        self,
        val_preds: np.ndarray,
        val_probs: np.ndarray,
        val_labels: np.ndarray,
        test_preds: np.ndarray,
        test_probs: np.ndarray,
        test_labels: np.ndarray,
    ) -> None:
        """Save predictions and metrics to CSV files.

        Args:
            val_preds: Validation predictions.
            val_probs: Validation probabilities.
            val_labels: Validation labels.
            test_preds: Test predictions.
            test_probs: Test probabilities.
            test_labels: Test labels.
        """
        results_dir = Path(self.config.get("outputs", "results_dir", default="outputs/results"))
        ensure_dir(str(results_dir))

        # Save validation predictions
        val_results = pd.DataFrame(
            {
                "true_label": val_labels,
                "prediction": val_preds,
                **{f"prob_class_{i}": val_probs[:, i] for i in range(val_probs.shape[1])},
            }
        )
        val_results.to_csv(results_dir / "lstm_val_predictions.csv", index=False)

        # Save test predictions
        test_results = pd.DataFrame(
            {
                "true_label": test_labels,
                "prediction": test_preds,
                **{f"prob_class_{i}": test_probs[:, i] for i in range(test_probs.shape[1])},
            }
        )
        test_results.to_csv(results_dir / "lstm_test_predictions.csv", index=False)

        # Save metrics history
        metrics_df = pd.DataFrame(self.metrics_history)
        metrics_df.to_csv(results_dir / "lstm_training_metrics.csv", index=False)

        # Compute and save final metrics
        val_metrics = compute_metrics(val_labels, val_preds)
        test_metrics = compute_metrics(test_labels, test_preds)

        metrics_summary = {
            "validation": val_metrics,
            "test": test_metrics,
        }

        with open(results_dir / "lstm_final_metrics.json", "w") as f:
            json.dump(metrics_summary, f, indent=2)

        print(f"\nResults saved to {results_dir}")
        print(f"Validation Metrics: {val_metrics}")
        print(f"Test Metrics: {test_metrics}")

    def run(self) -> None:
        """Run complete pipeline: data loading → training → evaluation → results."""
        print("=" * 80)
        print("LSTM PIPELINE")
        print("=" * 80)

        # Load data
        raw_data_path = self.config.get("data", "raw_path", default="data/raw/matches.csv")
        sequences, labels, feature_cols, match_dates = self.load_and_prepare_data(raw_data_path)
        self.input_size = sequences.shape[-1]

        # Create dataloaders
        train_loader, val_loader, test_loader = self.create_dataloaders(
            sequences, labels, match_dates
        )

        # Build model
        model = self.build_model()
        print(f"Model:\n{model}")

        # Train
        print("\nStarting training...")
        self.train(model, train_loader, val_loader)

        # Evaluate and predict
        print("\nGenerating predictions...")
        val_preds, val_probs, val_labels = self.predict(model, val_loader)
        test_preds, test_probs, test_labels = self.predict(model, test_loader)

        # Save results
        self.save_results(
            val_preds, val_probs, val_labels, test_preds, test_probs, test_labels
        )

        print("=" * 80)
        print("Pipeline complete!")
        print("=" * 80)
