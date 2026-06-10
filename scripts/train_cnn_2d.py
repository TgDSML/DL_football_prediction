"""
CNN training script for football match prediction.

Configured here for the 2D CNN model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn import build_cnn_model
from src.dataset_cnn import CNNDataModule
from src.evaluation.cnn_metrics import evaluate_and_save_metrics


MODEL_TYPE = "conv2d"
SEQUENCE_LENGTH = 5
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

        if "loss" in early_stopping_metric:
            self.best_metric = float("inf")
        else:
            self.best_metric = -float("inf")

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
        if model_type == "conv2d":
            return sequences.unsqueeze(1)
        raise ValueError(f"This script is configured for conv2d only, got: {model_type}")

    def train_epoch(self, train_loader: DataLoader, model_type: str):
        from sklearn.metrics import f1_score

        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc="Training", leave=False)
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
            for batch in tqdm(val_loader, desc="Validating", leave=False):
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
        if "loss" in self.early_stopping_metric:
            improved = current_metric < self.best_metric
        else:
            improved = current_metric > self.best_metric

        if improved:
            self.best_metric = current_metric
            self.patience_counter = 0
            return False

        self.patience_counter += 1
        return self.patience_counter >= self.early_stopping_patience

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None, model_type: str):
        import copy

        print(f"\nTraining {model_type} for {self.num_epochs} epochs...")
        print(f"Early stopping metric: {self.early_stopping_metric}")

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
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break
                else:
                    if self.patience_counter == 0:
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
            for batch in tqdm(test_loader, desc="Predicting", leave=False):
                sequences = batch["sequences"].to(self.device)
                labels = batch["labels"].to(self.device)

                sequences = self._prepare_inputs(sequences, model_type)

                logits = self.model(sequences)
                proba = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                all_preds.append(preds.cpu().numpy())
                all_proba.append(proba.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        return (
            np.concatenate(all_preds),
            np.concatenate(all_proba),
            np.concatenate(all_labels),
        )


def main():
    print("=" * 70)
    print("CNN Football Prediction - 2D CNN")
    print("=" * 70)
    print(f"Model type: {MODEL_TYPE}")
    print(f"Sequence length: k={SEQUENCE_LENGTH}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS detected")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    data_dir = PROJECT_ROOT / "data" / "processed"
    use_cnn_format = False

    print("\n[Step 1] Loading CNN data...")
    data_module = CNNDataModule(
        data_dir=data_dir,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        normalize=NORMALIZE,
        use_cnn_format=use_cnn_format,
    )
    data_module.setup()

    train_dataset = data_module.train_dataset
    val_dataset = data_module.val_dataset
    test_dataset = data_module.test_dataset

    if train_dataset is None:
        raise ValueError("Training dataset is None. Check dataset generation and paths.")

    print("\n[Step 2] Computing class weights...")
    train_labels = train_dataset.labels.cpu().numpy()
    num_classes = int(train_labels.max()) + 1
    class_counts = np.bincount(train_labels, minlength=num_classes)

    class_weights_numpy = np.zeros_like(class_counts, dtype=np.float32)
    nonzero_mask = class_counts > 0
    class_weights_numpy[nonzero_mask] = len(train_labels) / (num_classes * class_counts[nonzero_mask])
    class_weights = torch.tensor(class_weights_numpy, dtype=torch.float32).to(device)

    for i, count in enumerate(class_counts):
        pct = 100 * count / len(train_labels) if len(train_labels) > 0 else 0.0
        print(f"  Class {i}: {count:5d} ({pct:5.1f}%) weight={class_weights[i]:.4f}")

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

    print("\n[Step 3] Building 2D model...")
    _, seq_len, num_features = train_dataset.sequences.shape
    model = build_cnn_model(
        model_type="conv2d",
        input_height=seq_len,
        input_width=num_features,
        input_channels=1,
        num_classes=num_classes,
        num_filters=[32, 64, 128],
        kernel_size=3,
        dropout=DROPOUT,
    )
    print(f"Expected input after reshape: (N, 1, {seq_len}, {num_features})")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n[Step 4] Training...")
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

    history = trainer.fit(train_loader, val_loader, model_type=MODEL_TYPE)

    results_dir = PROJECT_ROOT / "artifacts" / "cnn_results" / f"{MODEL_TYPE}_k{SEQUENCE_LENGTH}"
    results_dir.mkdir(parents=True, exist_ok=True)

    if test_loader is not None:
        print("\n[Step 5] Evaluating on test set...")
        test_preds, test_proba, test_labels = trainer.predict(test_loader, model_type=MODEL_TYPE)

        evaluate_and_save_metrics(
            y_true=test_labels,
            y_pred=test_preds,
            y_proba=test_proba,
            output_dir=results_dir,
            prefix=f"{MODEL_TYPE}_k{SEQUENCE_LENGTH}_test",
        )
    else:
        print("\n[Step 5] No test set available.")

    print("\n[Step 6] Saving model and history...")
    model_dir = PROJECT_ROOT / "outputs" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{MODEL_TYPE}_k{SEQUENCE_LENGTH}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    max_len = max(len(history["train_losses"]), len(history["val_losses"]))
    history_rows = []
    for i in range(max_len):
        history_rows.append({
            "epoch": i + 1,
            "train_loss": history["train_losses"][i] if i < len(history["train_losses"]) else np.nan,
            "train_acc": history["train_accs"][i] if i < len(history["train_accs"]) else np.nan,
            "train_f1": history["train_f1s"][i] if i < len(history["train_f1s"]) else np.nan,
            "val_loss": history["val_losses"][i] if i < len(history["val_losses"]) else np.nan,
            "val_acc": history["val_accs"][i] if i < len(history["val_accs"]) else np.nan,
            "val_f1": history["val_f1s"][i] if i < len(history["val_f1s"]) else np.nan,
        })

    history_df = pd.DataFrame(history_rows)
    history_path = results_dir / "training_history.csv"
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to: {history_path}")

    summary_path = results_dir / "run_summary.csv"
    summary_df = pd.DataFrame([{
        "model_type": MODEL_TYPE,
        "sequence_length": SEQUENCE_LENGTH,
        "batch_size": BATCH_SIZE,
        "num_epochs_requested": NUM_EPOCHS,
        "epochs_trained": len(history["train_losses"]),
        "learning_rate": LEARNING_RATE,
        "dropout": DROPOUT,
        "use_focal_loss": USE_FOCAL_LOSS,
        "focal_gamma": FOCAL_GAMMA if USE_FOCAL_LOSS else np.nan,
        "label_smoothing": LABEL_SMOOTHING if not USE_FOCAL_LOSS else np.nan,
        "best_epoch": history.get("best_epoch", np.nan),
        "best_metric": history.get("best_metric", np.nan),
        "early_stopping_metric": EARLY_STOPPING_METRIC,
        "model_path": str(model_path),
    }])
    summary_df.to_csv(summary_path, index=False)
    print(f"Run summary saved to: {summary_path}")

    print("\n" + "=" * 70)
    print("Training complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()