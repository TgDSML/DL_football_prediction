"""
CNN evaluation utilities for football match prediction.

Computes key multiclass metrics and saves them for experiment tracking.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


class CNNMetricsCollector:
    CLASS_NAMES = {0: "Away Win", 1: "Draw", 2: "Home Win"}

    def __init__(self, output_dir: str | Path | None = None):
        if output_dir is None:
            output_dir = Path(__file__).resolve().parents[2] / "artifacts" / "cnn_results"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics: dict = {}

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray | None = None,
    ) -> dict:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        self.metrics = {}

        overall = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }

        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        overall.update(
            {
                "weighted_precision": float(prec_w),
                "weighted_recall": float(rec_w),
                "weighted_f1": float(f1_w),
                "macro_precision": float(prec_m),
                "macro_recall": float(rec_m),
                "macro_f1": float(f1_m),
                "total_samples": int(len(y_true)),
            }
        )
        self.metrics["overall"] = overall

        p, r, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1, 2], zero_division=0
        )

        per_class = {}
        for idx, class_name in self.CLASS_NAMES.items():
            mask = y_true == idx
            if mask.sum() == 0:
                continue
            per_class[class_name] = {
                "precision": float(p[idx]),
                "recall": float(r[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
                "accuracy": float((y_pred[mask] == y_true[mask]).mean()),
            }
        self.metrics["per_class"] = per_class

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        self.metrics["confusion_matrix"] = {
            "true_away_win": {
                "pred_away_win": int(cm[0, 0]),
                "pred_draw": int(cm[0, 1]),
                "pred_home_win": int(cm[0, 2]),
            },
            "true_draw": {
                "pred_away_win": int(cm[1, 0]),
                "pred_draw": int(cm[1, 1]),
                "pred_home_win": int(cm[1, 2]),
            },
            "true_home_win": {
                "pred_away_win": int(cm[2, 0]),
                "pred_draw": int(cm[2, 1]),
                "pred_home_win": int(cm[2, 2]),
            },
        }

        self.metrics["classification_report"] = classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=[self.CLASS_NAMES[i] for i in [0, 1, 2]],
            zero_division=0,
            output_dict=True,
        )

        if y_proba is not None:
            y_proba = np.asarray(y_proba)
            self.metrics["probability_stats"] = {
                "mean_max_probability": float(np.mean(np.max(y_proba, axis=1))),
                "mean_entropy": float(np.mean(-np.sum(y_proba * np.log(y_proba + 1e-10), axis=1))),
                "confidence_std": float(np.std(np.max(y_proba, axis=1))),
            }

        return self.metrics

    def save_metrics_json(self, filename: str | None = None) -> Path:
        if filename is None:
            filename = f"cnn_metrics_{self.timestamp}.json"
        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)
        return path

    def save_metrics_csv(self, filename: str | None = None) -> Path:
        if filename is None:
            filename = f"cnn_metrics_{self.timestamp}.csv"
        path = self.output_dir / filename

        flat = {}
        for k, v in self.metrics.get("overall", {}).items():
            flat[f"overall_{k}"] = v

        for class_name, vals in self.metrics.get("per_class", {}).items():
            for k, v in vals.items():
                flat[f"{class_name}_{k}"] = v

        if "probability_stats" in self.metrics:
            for k, v in self.metrics["probability_stats"].items():
                flat[f"probability_{k}"] = v

        pd.DataFrame([flat]).to_csv(path, index=False)
        return path

    def save_confusion_matrix_csv(self, filename: str | None = None) -> Path:
        if filename is None:
            filename = f"confusion_matrix_{self.timestamp}.csv"
        path = self.output_dir / filename

        cm = self.metrics.get("confusion_matrix", {})
        pd.DataFrame(cm).T.to_csv(path)
        return path

    def save_report(self, filename: str | None = None) -> Path:
        if filename is None:
            filename = f"cnn_report_{self.timestamp}.txt"
        path = self.output_dir / filename

        lines = []
        lines.append("=" * 70)
        lines.append("CNN MODEL EVALUATION REPORT")
        lines.append("=" * 70)
        lines.append(f"Timestamp: {self.timestamp}")
        lines.append("")

        lines.append("OVERALL METRICS")
        lines.append("-" * 70)
        for k, v in self.metrics.get("overall", {}).items():
            lines.append(f"{k:<30} {v:.4f}" if isinstance(v, float) else f"{k:<30} {v}")

        lines.append("")
        lines.append("PER-CLASS METRICS")
        lines.append("-" * 70)
        for class_name, vals in self.metrics.get("per_class", {}).items():
            lines.append(f"{class_name}:")
            for k, v in vals.items():
                lines.append(f"  {k:<26} {v:.4f}" if isinstance(v, float) else f"  {k:<26} {v}")
            lines.append("")

        lines.append("CONFUSION MATRIX")
        lines.append("-" * 70)
        lines.append("               pred_away  pred_draw  pred_home")
        cm = self.metrics.get("confusion_matrix", {})
        for true_name, row in cm.items():
            lines.append(
                f"{true_name:<15} {row.get('pred_away_win', 0):>10} {row.get('pred_draw', 0):>10} {row.get('pred_home_win', 0):>10}"
            )

        if "probability_stats" in self.metrics:
            lines.append("")
            lines.append("PROBABILITY STATISTICS")
            lines.append("-" * 70)
            for k, v in self.metrics["probability_stats"].items():
                lines.append(f"{k:<30} {v:.4f}")

        lines.append("=" * 70)

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def save_all(self, prefix: str = "cnn") -> dict:
        return {
            "json": self.save_metrics_json(f"{prefix}_metrics_{self.timestamp}.json"),
            "csv": self.save_metrics_csv(f"{prefix}_metrics_{self.timestamp}.csv"),
            "confusion_matrix": self.save_confusion_matrix_csv(f"{prefix}_cm_{self.timestamp}.csv"),
            "report": self.save_report(f"{prefix}_report_{self.timestamp}.txt"),
        }

    def print_summary(self):
        overall = self.metrics.get("overall", {})
        print("\n" + "=" * 70)
        print("CNN EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Accuracy:   {overall.get('accuracy', 0):.4f}")
        print(f"Weighted F1:{overall.get('weighted_f1', 0):.4f}")
        print(f"Macro F1:   {overall.get('macro_f1', 0):.4f}")
        print(f"Samples:    {overall.get('total_samples', 0)}")
        print("=" * 70)


def evaluate_and_save_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    output_dir: str | Path | None = None,
    prefix: str = "cnn",
) -> CNNMetricsCollector:
    collector = CNNMetricsCollector(output_dir=output_dir)
    collector.compute_metrics(y_true, y_pred, y_proba)
    collector.save_all(prefix=prefix)
    collector.print_summary()
    return collector