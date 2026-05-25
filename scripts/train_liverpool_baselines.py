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
from sklearn.svm import SVC
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = PROJECT_ROOT / "data/processed/team_centric_features.csv"
REPORT_PATH = PROJECT_ROOT / "outputs/reports/liverpool_baseline_report.txt"
PREDICTIONS_PATH = PROJECT_ROOT / "outputs/results/liverpool_baseline_predictions.csv"

TEAM = "Liverpool"
TEST_FRACTION = 0.20
TARGET_NAMES = ["win", "draw", "loss"]

NUMERIC_FEATURES = [
    "is_home",
    "rest_days",
    "match_month",
    "points_last_5",
    "wins_last_5",
    "draws_last_5",
    "losses_last_5",
    "goals_for_last_5",
    "goals_against_last_5",
    "goal_diff_last_5",
    "avg_goals_for_last_5",
    "avg_goals_against_last_5",
    "clean_sheets_last_5",
    "failed_to_score_last_5",
    "avg_shots_last_5",
    "avg_shots_on_target_last_5",
    "avg_corners_last_5",
    "avg_yellow_cards_last_5",
    "avg_red_cards_last_5",
]
CATEGORICAL_FEATURES = ["opponent"]
FEATURES = [*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]


def load_liverpool_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run scripts/build_features.py first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    liverpool = df[df["team"] == TEAM].copy()
    if liverpool.empty:
        raise ValueError(f"No rows found for team={TEAM}")

    missing_features = [col for col in FEATURES if col not in liverpool.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    return liverpool.sort_values(["Date", "match_id"]).reset_index(drop=True)


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_size = max(1, int(round(len(df) * TEST_FRACTION)))
    split_index = len(df) - test_size
    if split_index <= 0:
        raise ValueError("Not enough Liverpool rows for chronological split")

    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()
    return train, test


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
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
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.0,
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "dummy_most_frequent": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(scale_numeric=False)),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(scale_numeric=False)),
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
                ("preprocess", make_preprocessor(scale_numeric=False)),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
        "svm_rbf": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(scale_numeric=True)),
                (
                    "model",
                    SVC(
                        C=1.0,
                        kernel="rbf",
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "xgboost": Pipeline(
            steps=[
                ("preprocess", make_preprocessor(scale_numeric=False)),
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


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
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


def target_distribution(df: pd.DataFrame) -> str:
    distribution = (
        df["target"]
        .value_counts()
        .sort_index()
        .rename(index={0: "0 win", 1: "1 draw", 2: "2 loss"})
    )
    return distribution.to_string()


def build_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    results: dict[str, dict[str, object]],
) -> str:
    lines = [
        "Liverpool Starter ML Baselines",
        "=" * 36,
        f"Dataset: {DATA_PATH.relative_to(PROJECT_ROOT)}",
        f"Team: {TEAM}",
        f"Rows: train={len(train)}, test={len(test)}",
        f"Train date range: {train['Date'].min().date()} to {train['Date'].max().date()}",
        f"Test date range: {test['Date'].min().date()} to {test['Date'].max().date()}",
        "",
        "Feature Set",
        "=" * 36,
        "\n".join(FEATURES),
        "",
        "Target Encoding",
        "=" * 36,
        "0 = Liverpool win",
        "1 = draw",
        "2 = Liverpool loss",
        "",
        "Target Distribution",
        "=" * 36,
        "[train]",
        target_distribution(train),
        "",
        "[test]",
        target_distribution(test),
        "",
        "Results",
        "=" * 36,
    ]

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
    data = load_liverpool_data()
    train, test = chronological_split(data)
    x_train = train[FEATURES]
    y_train = train["target"]
    x_test = test[FEATURES]
    y_test = test["target"]

    results = {}
    prediction_output = test[["Date", "team", "opponent", "is_home", "target"]].copy()

    for model_name, model in build_models().items():
        model.fit(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)
        results[model_name] = metrics
        prediction_output[f"{model_name}_prediction"] = metrics["predictions"]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(train, test, results), encoding="utf-8")
    prediction_output.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Trained Liverpool baselines on {len(train)} older matches")
    print(f"Evaluated on {len(test)} latest matches")
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
