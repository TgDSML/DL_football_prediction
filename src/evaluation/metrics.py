from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def summarize_metrics(metrics: dict[str, float]) -> str:
    if not metrics:
        return "No metrics available yet."
    return ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
