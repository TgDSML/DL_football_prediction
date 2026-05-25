from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_utils import (  # noqa: E402
    PROCESSED_SPLIT_DIR,
    TARGET_COLUMN,
    TARGET_NAMES,
    infer_feature_columns,
    load_processed_splits,
    split_feature_types,
    target_distribution,
    validate_processed_splits,
)


REPORT_PATH = PROJECT_ROOT / "outputs/reports/final_baseline_report.txt"
PREDICTIONS_PATH = PROJECT_ROOT / "outputs/results/final_baseline_predictions.csv"


def make_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        sparse_threshold=0.0,
    )


def build_models(
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Pipeline]:
    return {
        "dummy_most_frequent": Pipeline(
            steps=[
                (
                    "preprocess",
                    make_preprocessor(
                        numeric_features,
                        categorical_features,
                        scale_numeric=False,
                    ),
                ),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                (
                    "preprocess",
                    make_preprocessor(
                        numeric_features,
                        categorical_features,
                        scale_numeric=True,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                (
                    "preprocess",
                    make_preprocessor(
                        numeric_features,
                        categorical_features,
                        scale_numeric=False,
                    ),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            steps=[
                (
                    "preprocess",
                    make_preprocessor(
                        numeric_features,
                        categorical_features,
                        scale_numeric=False,
                    ),
                ),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                (
                    "preprocess",
                    make_preprocessor(
                        numeric_features,
                        categorical_features,
                        scale_numeric=False,
                    ),
                ),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, object]:
    predictions = model.predict(x_test)
    metrics: dict[str, object] = {
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1, 2]),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2],
            target_names=TARGET_NAMES,
            zero_division=0,
        ),
        "predictions": predictions,
    }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)
        metrics["log_loss"] = log_loss(y_test, probabilities, labels=[0, 1, 2])

    return metrics


def build_report(
    splits: dict[str, pd.DataFrame],
    feature_columns: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    results: dict[str, dict[str, object]],
) -> str:
    lines = [
        "Final Leakage-Free ML Baselines",
        "=" * 40,
        f"Dataset: {PROCESSED_SPLIT_DIR.relative_to(PROJECT_ROOT)}",
        "Feature source: src/features/build_team_features.py",
        "Training split: train",
        "Model selection split: val",
        "Final evaluation split: test",
        "",
        "Leakage Policy",
        "=" * 40,
        "Current-match statistics are used only to build shifted historical "
        "rolling features.",
        "They are never direct inputs for the target match.",
        "",
        "Split Sizes",
        "=" * 40,
    ]

    for split_name, df in splits.items():
        lines.append(
            f"{split_name}: {len(df)} rows, "
            f"{df['Date'].min().date()} to {df['Date'].max().date()}"
        )

    lines.extend(["", "Target Distribution", "=" * 40])
    for split_name, df in splits.items():
        lines.extend([f"[{split_name}]", target_distribution(df), ""])

    lines.extend(
        [
            "Feature Columns",
            "=" * 40,
            f"Total: {len(feature_columns)}",
            "",
            "Numeric",
            "-" * 40,
            "\n".join(numeric_features),
            "",
            "Categorical",
            "-" * 40,
            "\n".join(categorical_features),
            "",
            "Results On Test Split",
            "=" * 40,
        ]
    )

    for model_name, metrics in results.items():
        lines.extend(
            [
                f"[{model_name}]",
                f"accuracy: {metrics['accuracy']:.4f}",
                f"balanced_accuracy: {metrics['balanced_accuracy']:.4f}",
                f"macro_f1: {metrics['macro_f1']:.4f}",
            ]
        )
        if "log_loss" in metrics:
            lines.append(f"log_loss: {metrics['log_loss']:.4f}")
        lines.extend(
            [
                "confusion_matrix rows=true cols=pred labels=[win, draw, loss]",
                str(metrics["confusion_matrix"]),
                "classification_report:",
                str(metrics["classification_report"]),
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    splits = load_processed_splits()
    feature_columns = infer_feature_columns(splits["train"])
    validate_processed_splits(splits, feature_columns)
    numeric_features, categorical_features = split_feature_types(feature_columns)

    x_train = splits["train"][feature_columns]
    y_train = splits["train"][TARGET_COLUMN]
    x_test = splits["test"][feature_columns]
    y_test = splits["test"][TARGET_COLUMN]

    results = {}
    prediction_output = splits["test"][
        ["match_id", "Date", "team", "opponent", "is_home", TARGET_COLUMN]
    ].copy()

    for model_name, model in build_models(numeric_features, categorical_features).items():
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)
        results[model_name] = metrics
        prediction_output[f"{model_name}_prediction"] = metrics["predictions"]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        build_report(
            splits,
            feature_columns,
            numeric_features,
            categorical_features,
            results,
        ),
        encoding="utf-8",
    )
    prediction_output.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Loaded processed splits from {PROCESSED_SPLIT_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Feature columns: {len(feature_columns)}")
    print(f"Saved report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved predictions: {PREDICTIONS_PATH.relative_to(PROJECT_ROOT)}")
    for model_name, metrics in results.items():
        print(
            f"{model_name}: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
