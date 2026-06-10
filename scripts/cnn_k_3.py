from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn import build_cnn_model
from src.dataset_cnn import CNNDataModule
from src.evaluation.cnn_metrics import evaluate_and_save_metrics

SEQUENCE_LENGTH = 3
MODEL_TYPES = ["conv1d", "conv2d", "hybrid"]

BATCH_SIZE = 32
NUM_EPOCHS = 60
LEARNING_RATE = 5e-4
DROPOUT = 0.5
NUM_WORKERS = 0
NORMALIZE = True
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 3.5
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_METRIC = "f1_weighted"
LABEL_SMOOTHING = 0.1


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = torch.nn.functional.cross_entropy(
            logits,
            target,
            reduction="none",
            weight=self.alpha,
        )
        p = torch.softmax(logits, dim=1)
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class CNNTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        num_epochs: int = 50,
        class_weights: torch.Tensor | None = None,
        use_focal_loss: bool = False,
        focal_gamma: float = 3.0,
        early_stopping_patience: int = 12,
        early_stopping_metric: str = "f1_weighted",
        label_smoothing: float = 0.1,
    ):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric

        if use_focal_loss:
            self.criterion = FocalLoss(alpha=class_weights, gamma=focal_gamma)
            print(f"Using FocalLoss(gamma={focal_gamma})")
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing,
            )
            print(f"Using CrossEntropyLoss(label_smoothing={label_smoothing})")

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        self.best_metric = float("inf") if "loss" in early_stopping_metric else -float("inf")
        self.patience_counter = 0
        self.best_epoch = 0
        self.best_state_dict = None

        self.train_losses = []
        self.train_accs = []
        self.train_f1s = []
        self.val_losses = []
        self.val_accs = []
        self.val_f1s = []

    def _prepare_inputs(self, sequences: torch.Tensor, model_type: str) -> torch.Tensor:
        if model_type == "conv1d":
            return sequences
        if model_type == "conv2d":
            return sequences.unsqueeze(1)
        if model_type == "hybrid":
            return sequences
        raise ValueError(f"Unknown model_type: {model_type}")

    def train_epoch(self, train_loader: DataLoader, model_type: str):
        from sklearn.metrics import f1_score

        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc=f"Training {model_type}", leave=False)
        for batch in pbar:
            sequences = batch["sequences"].to(self.device)
            labels = batch["labels"].to(self.device)
            sequences = self._prepare_inputs(sequences, model_type)

            self.optimizer.zero_grad()
            logits = self.model(sequences)
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += batch_size

            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

            pbar.set_postfix({
                "loss": f"{total_loss / max(total, 1):.4f}",
                "acc": f"{correct / max(total, 1):.4f}",
            })

        avg_loss = total_loss / max(total, 1)
        avg_acc = correct / max(total, 1)
        avg_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        return avg_loss, avg_acc, avg_f1

    def validate(self, val_loader: DataLoader | None, model_type: str):
        from sklearn.metrics import f1_score

        if val_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating {model_type}", leave=False):
                sequences = batch["sequences"].to(self.device)
                labels = batch["labels"].to(self.device)
                sequences = self._prepare_inputs(sequences, model_type)

                logits = self.model(sequences)
                loss = self.criterion(logits, labels)

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += batch_size

                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

        avg_loss = total_loss / max(total, 1)
        avg_acc = correct / max(total, 1)
        avg_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        return avg_loss, avg_acc, avg_f1

    def should_stop_early(self, current_metric: float) -> bool:
        improved = current_metric < self.best_metric if "loss" in self.early_stopping_metric else current_metric > self.best_metric
        if improved:
            self.best_metric = current_metric
            self.patience_counter = 0
            return False
        self.patience_counter += 1
        return self.patience_counter >= self.early_stopping_patience

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None, model_type: str):
        import copy

        print(f"\nTraining {model_type} for {self.num_epochs} epochs...")
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            train_loss, train_acc, train_f1 = self.train_epoch(train_loader, model_type)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.train_f1s.append(train_f1)
            print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")

            if val_loader is not None:
                val_loss, val_acc, val_f1 = self.validate(val_loader, model_type)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
                self.val_f1s.append(val_f1)
                print(f"  Val Loss:   {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

                self.scheduler.step(val_loss)

                if self.early_stopping_metric == "f1_weighted":
                    current_metric = val_f1
                elif self.early_stopping_metric == "accuracy":
                    current_metric = val_acc
                else:
                    current_metric = val_loss

                if self.should_stop_early(current_metric):
                    print(f"Early stopping triggered for {model_type} at epoch {epoch + 1}")
                    break
                elif self.patience_counter == 0:
                    self.best_epoch = epoch + 1
                    self.best_state_dict = copy.deepcopy(self.model.state_dict())

        if val_loader is None:
            self.best_state_dict = self.model.state_dict()
            self.best_epoch = len(self.train_losses)

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        return {
            "train_losses": self.train_losses,
            "train_accs": self.train_accs,
            "train_f1s": self.train_f1s,
            "val_losses": self.val_losses,
            "val_accs": self.val_accs,
            "val_f1s": self.val_f1s,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
        }

    def predict(self, test_loader: DataLoader, model_type: str):
        self.model.eval()
        all_preds = []
        all_proba = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Predicting {model_type}", leave=False):
                sequences = batch["sequences"].to(self.device)
                labels = batch["labels"].to(self.device)
                sequences = self._prepare_inputs(sequences, model_type)

                logits = self.model(sequences)
                proba = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                all_preds.append(preds.cpu().numpy())
                all_proba.append(proba.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        return np.concatenate(all_preds), np.concatenate(all_proba), np.concatenate(all_labels)


def compute_classification_summary(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    wp, wr, wf1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    mp, mr, mf1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": acc,
        "weighted_precision": wp,
        "weighted_recall": wr,
        "weighted_f1": wf1,
        "macro_precision": mp,
        "macro_recall": mr,
        "macro_f1": mf1,
    }


def build_model_for_type(model_type: str, train_dataset, num_classes: int, dropout: float):
    if model_type == "conv1d":
        input_channels = train_dataset.sequences.shape[1]
        return build_cnn_model(
            model_type="conv1d",
            input_channels=input_channels,
            num_classes=num_classes,
            num_filters=[64, 128, 64],
            kernel_size=3,
            dropout=dropout,
        )

    if model_type == "conv2d":
        _, seq_len, num_features = train_dataset.sequences.shape
        return build_cnn_model(
            model_type="conv2d",
            input_height=seq_len,
            input_width=num_features,
            input_channels=1,
            num_classes=num_classes,
            num_filters=[32, 64, 128],
            kernel_size=3,
            dropout=dropout,
        )

    if model_type == "hybrid":
        _, _, num_features = train_dataset.sequences.shape
        return build_cnn_model(
            model_type="hybrid",
            input_channels=num_features,
            num_classes=num_classes,
            conv1d_filters=[64, 128],
            conv2d_filters=[32, 64],
            dropout=dropout,
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def create_data_module(model_type: str):
    use_cnn_format = model_type == "conv1d"
    data_dir = PROJECT_ROOT / "data" / "processed"
    data_module = CNNDataModule(
        data_dir=data_dir,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        normalize=NORMALIZE,
        use_cnn_format=use_cnn_format,
    )
    data_module.setup()
    return data_module


def run_single_experiment(model_type: str, device: torch.device):
    print("\n" + "=" * 80)
    print(f"RUNNING MODEL: {model_type} | k={SEQUENCE_LENGTH}")
    print("=" * 80)

    data_module = create_data_module(model_type)
    train_dataset = data_module.train_dataset
    val_dataset = data_module.val_dataset
    test_dataset = data_module.test_dataset

    if train_dataset is None:
        raise ValueError("Training dataset is None")

    train_labels = train_dataset.labels.cpu().numpy()
    num_classes = int(train_labels.max()) + 1
    class_counts = np.bincount(train_labels, minlength=num_classes)

    class_weights_numpy = np.zeros_like(class_counts, dtype=np.float32)
    nonzero_mask = class_counts > 0
    class_weights_numpy[nonzero_mask] = len(train_labels) / (num_classes * class_counts[nonzero_mask])
    class_weights = torch.tensor(class_weights_numpy, dtype=torch.float32).to(device)

    sample_weights = class_weights.cpu().numpy()[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(train_labels),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        collate_fn=data_module.collate_fn,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=data_module.collate_fn,
        )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=data_module.collate_fn,
        )

    model = build_model_for_type(model_type, train_dataset, num_classes, DROPOUT)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    trainer = CNNTrainer(
        model=model,
        device=device,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        class_weights=class_weights,
        use_focal_loss=USE_FOCAL_LOSS,
        focal_gamma=FOCAL_GAMMA,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_metric=EARLY_STOPPING_METRIC,
        label_smoothing=LABEL_SMOOTHING,
    )

    history = trainer.fit(train_loader, val_loader, model_type=model_type)

    results_dir = PROJECT_ROOT / "artifacts" / "cnn_results" / f"{model_type}_k{SEQUENCE_LENGTH}"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_dir = PROJECT_ROOT / "outputs" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_type}_k{SEQUENCE_LENGTH}.pth"
    torch.save(trainer.model.state_dict(), model_path)

    history_df = pd.DataFrame({
        "epoch": np.arange(1, len(history["train_losses"]) + 1),
        "train_loss": history["train_losses"],
        "train_acc": history["train_accs"],
        "train_f1": history["train_f1s"],
        "val_loss": history["val_losses"] + [np.nan] * max(0, len(history["train_losses"]) - len(history["val_losses"])),
        "val_acc": history["val_accs"] + [np.nan] * max(0, len(history["train_losses"]) - len(history["val_accs"])),
        "val_f1": history["val_f1s"] + [np.nan] * max(0, len(history["train_losses"]) - len(history["val_f1s"])),
    })
    history_df.to_csv(results_dir / "training_history.csv", index=False)

    result_row = {
        "model_type": model_type,
        "sequence_length": SEQUENCE_LENGTH,
        "best_epoch": history.get("best_epoch", np.nan),
        "best_metric": history.get("best_metric", np.nan),
        "model_path": str(model_path),
        "total_params": total_params,
        "trainable_params": trainable_params,
    }

    if test_loader is not None:
        test_preds, test_proba, test_labels = trainer.predict(test_loader, model_type=model_type)
        evaluate_and_save_metrics(
            y_true=test_labels,
            y_pred=test_preds,
            y_proba=test_proba,
            output_dir=results_dir,
            prefix=f"{model_type}_k{SEQUENCE_LENGTH}_test",
        )
        result_row.update(compute_classification_summary(test_labels, test_preds))

    del trainer, model, data_module, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result_row


def main():
    print("=" * 80)
    print("CNN k=3 EXPERIMENTS: 1D vs 2D vs HYBRID")
    print("=" * 80)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    comparison_rows = []
    for model_type in MODEL_TYPES:
        comparison_rows.append(run_single_experiment(model_type, device))

    comparison_df = pd.DataFrame(comparison_rows).sort_values(by="weighted_f1", ascending=False)
    comparison_dir = PROJECT_ROOT / "artifacts" / "cnn_results"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = comparison_dir / "cnn_k3_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(comparison_df[[
        "model_type",
        "sequence_length",
        "accuracy",
        "weighted_f1",
        "macro_f1",
        "best_epoch",
        "best_metric",
    ]].to_string(index=False))
    print(f"\nSaved comparison to: {comparison_path}")


if __name__ == "__main__":
    main()