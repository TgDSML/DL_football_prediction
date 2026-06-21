from __future__ import annotations

import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_SPLIT_DIR = Path(PROJECT_ROOT) / "data" / "raw" / "splits"
ARTIFACT_DIR = Path(PROJECT_ROOT) / "artifacts" / "ml_artifacts"

TARGET_MAP = {"H": 0, "D": 1, "A": 2}
TARGET_NAMES = ["HomeWin", "Draw", "AwayWin"]

LEAKAGE_COLUMNS = {
    "FTR",
    "FTHG",
    "FTAG",
    "HTHG",
    "HTAG",
    "HTR",
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HC",
    "AC",
    "HY",
    "AY",
    "HR",
    "AR",
}

SAFE_BASE_COLUMNS = {
    "Date",
    "HomeTeam",
    "AwayTeam",
    "Referee",
    "season",
    "Season",
}

BOOKMAKER_PREFIXES = (
    "B365",
    "BW",
    "IW",
    "WH",
    "VC",
    "LB",
    "PS",
    "Max",
    "Avg",
)
ODDS_SUFFIXES = ("H", "D", "A", "CH", "CD", "CA")


def load_data() -> dict[str, pd.DataFrame]:
    splits: dict[str, pd.DataFrame] = {}
    missing = []
    for split_name in ("train", "val", "test"):
        path = RAW_SPLIT_DIR / f"{split_name}.csv"
        if not path.exists():
            missing.append(path)
            continue
        splits[split_name] = pd.read_csv(path)

    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing raw split file(s). Run `python data/split_data.py` first:\n"
            f"{missing_text}"
        )

    return splits


def build_target(df: pd.DataFrame) -> pd.Series:
    if "FTR" not in df.columns:
        raise ValueError("Raw split is missing required target column `FTR`.")

    y = df["FTR"].map(TARGET_MAP)
    bad_targets = df.loc[y.isna(), "FTR"].dropna().unique().tolist()
    if bad_targets:
        raise ValueError(f"Unexpected FTR values: {bad_targets}")
    if y.isna().any():
        raise ValueError("FTR contains missing values.")
    return y.astype("int64")


def _is_odds_column(column: str) -> bool:
    return any(
        column == f"{prefix}{suffix}"
        for prefix in BOOKMAKER_PREFIXES
        for suffix in ODDS_SUFFIXES
    )


def select_safe_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    selected_raw = [
        col
        for col in df.columns
        if col in SAFE_BASE_COLUMNS or _is_odds_column(col)
    ]
    leakage_excluded = [col for col in df.columns if col in LEAKAGE_COLUMNS]
    unknown_excluded = [
        col
        for col in df.columns
        if col not in selected_raw and col not in leakage_excluded
    ]

    X = df[selected_raw].copy()
    if "Date" in X.columns:
        dates = pd.to_datetime(X["Date"], errors="coerce", dayfirst=False)
        if dates.isna().any():
            fallback = pd.to_datetime(X["Date"], errors="coerce", dayfirst=True)
            dates = dates.fillna(fallback)
        X["match_year"] = dates.dt.year
        X["match_month"] = dates.dt.month
        X["match_dayofweek"] = dates.dt.dayofweek
        X = X.drop(columns=["Date"])

    used_columns = X.columns.tolist()
    excluded_columns = leakage_excluded + unknown_excluded
    if not used_columns:
        raise ValueError("No leakage-safe raw pre-match features were selected.")
    return X, used_columns, excluded_columns


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers = []
    if numeric_features:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_features))
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _one_hot_encoder()),
                    ]
                ),
                categorical_features,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_models(X_train: pd.DataFrame) -> dict[str, Pipeline]:
    return {
        "dummy_most_frequent": Pipeline(
            [
                ("preprocess", build_preprocessor(X_train, scale_numeric=False)),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("preprocess", build_preprocessor(X_train, scale_numeric=True)),
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
            [
                ("preprocess", build_preprocessor(X_train, scale_numeric=False)),
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
    }


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, object]:
    predictions = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro")),
        "classification_report": classification_report(
            y,
            predictions,
            labels=[0, 1, 2],
            target_names=TARGET_NAMES,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1, 2]),
        "predictions": predictions,
    }


def prediction_frame(
    raw_df: pd.DataFrame,
    y: pd.Series,
    all_results: dict[str, dict[str, dict[str, object]]],
    split_name: str,
) -> pd.DataFrame:
    metadata_cols = [col for col in ["Date", "season", "Season", "HomeTeam", "AwayTeam"] if col in raw_df.columns]
    out = raw_df[metadata_cols].copy()
    out["target"] = y.to_numpy()
    out["target_label"] = [TARGET_NAMES[label] for label in y]
    for model_name, split_results in all_results.items():
        out[f"{model_name}_prediction"] = split_results[split_name]["predictions"]
    return out


def save_report(
    used_columns: list[str],
    excluded_columns: list[str],
    results: dict[str, dict[str, dict[str, object]]],
    best_model_name: str,
) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "Raw Statistical Baseline",
        "=" * 40,
        "Data source: data/raw/splits/train.csv, val.csv, test.csv",
        "Methodology: raw pre-match tabular baseline before feature engineering.",
        "Target: 0=HomeWin, 1=Draw, 2=AwayWin",
        "",
        "Selected Safe Feature Columns",
        "=" * 40,
        "\n".join(used_columns),
        "",
        "Excluded Columns",
        "=" * 40,
        "\n".join(excluded_columns) if excluded_columns else "none",
        "",
        "Validation Selection",
        "=" * 40,
        f"Best model by validation macro F1: {best_model_name}",
        "",
        "Metrics",
        "=" * 40,
    ]

    summary_rows = []
    for model_name, split_results in results.items():
        for split_name in ("val", "test"):
            metrics = split_results[split_name]
            summary_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                }
            )
            lines.extend(
                [
                    f"[{model_name} - {split_name}]",
                    f"accuracy: {metrics['accuracy']:.4f}",
                    f"macro_f1: {metrics['macro_f1']:.4f}",
                    "confusion_matrix rows=true cols=pred labels=[HomeWin, Draw, AwayWin]",
                    np.array2string(np.asarray(metrics["confusion_matrix"])),
                    "classification_report:",
                    str(metrics["classification_report"]).rstrip(),
                    "",
                ]
            )

    (ARTIFACT_DIR / "raw_statistical_baseline_report.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    pd.DataFrame(summary_rows).to_csv(
        ARTIFACT_DIR / "raw_statistical_baseline_summary.csv",
        index=False,
    )


def main() -> None:
    splits = load_data()
    y_train = build_target(splits["train"])
    y_val = build_target(splits["val"])
    y_test = build_target(splits["test"])

    X_train, used_columns, excluded_columns = select_safe_features(splits["train"])
    X_val, val_used_columns, _ = select_safe_features(splits["val"])
    X_test, test_used_columns, _ = select_safe_features(splits["test"])

    if used_columns != val_used_columns or used_columns != test_used_columns:
        missing_val = sorted(set(used_columns).difference(val_used_columns))
        missing_test = sorted(set(used_columns).difference(test_used_columns))
        raise ValueError(
            "Safe feature columns differ across splits. "
            f"Missing in val: {missing_val}; missing in test: {missing_test}"
        )

    print("Selected leakage-safe feature columns:")
    for column in used_columns:
        print(f"  - {column}")
    print("\nExcluded leakage/unsafe columns:")
    for column in excluded_columns:
        print(f"  - {column}")

    models = build_models(X_train)
    results: dict[str, dict[str, dict[str, object]]] = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        results[model_name] = {
            "val": evaluate_model(model, X_val, y_val),
            "test": evaluate_model(model, X_test, y_test),
        }

    best_model_name = max(
        results,
        key=lambda name: float(results[name]["val"]["macro_f1"]),
    )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_frame(splits["val"], y_val, results, "val").to_csv(
        ARTIFACT_DIR / "raw_statistical_baseline_predictions_val.csv",
        index=False,
    )
    prediction_frame(splits["test"], y_test, results, "test").to_csv(
        ARTIFACT_DIR / "raw_statistical_baseline_predictions_test.csv",
        index=False,
    )
    with (ARTIFACT_DIR / "raw_statistical_baseline_best_model.pkl").open("wb") as file:
        pickle.dump(models[best_model_name], file)

    save_report(used_columns, excluded_columns, results, best_model_name)

    print(f"\nBest model by validation macro F1: {best_model_name}")
    print(f"Validation macro F1: {results[best_model_name]['val']['macro_f1']:.4f}")
    print(f"Test accuracy: {results[best_model_name]['test']['accuracy']:.4f}")
    print(f"Test macro F1: {results[best_model_name]['test']['macro_f1']:.4f}")
    print(f"Saved artifacts to: {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
