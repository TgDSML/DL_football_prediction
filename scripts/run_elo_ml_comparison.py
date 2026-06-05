from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ml_baselines import build_models, evaluate_model
from src.utils.data_utils import (
    FIXTURE_TARGET_NAMES,
    TARGET_COLUMN,
    infer_fixture_feature_columns,
    load_fixture_level_splits,
    split_fixture_feature_types,
    validate_fixture_level_splits,
)


REPORT_PATH = PROJECT_ROOT / "outputs/reports/elo_ml_comparison.txt"
MODEL_NAMES = ("logistic_regression", "random_forest", "xgboost")
ELO_COLUMNS = ["home_elo", "away_elo", "elo_diff"]


def evaluate_feature_set(
    splits: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> dict[str, dict[str, object]]:
    numeric_features, categorical_features = split_fixture_feature_types(feature_columns)
    models = build_models(numeric_features, categorical_features)
    x_train = splits["train"][feature_columns]
    y_train = splits["train"][TARGET_COLUMN]
    x_test = splits["test"][feature_columns]
    y_test = splits["test"][TARGET_COLUMN]

    results = {}
    for model_name in MODEL_NAMES:
        model = models[model_name]
        model.fit(x_train, y_train)
        results[model_name] = evaluate_model(
            model,
            x_test,
            y_test,
            FIXTURE_TARGET_NAMES,
        )
    return results


def metric_row(
    model_name: str,
    without_elo: dict[str, object],
    with_elo: dict[str, object],
) -> str:
    return (
        f"| {model_name} | {without_elo['accuracy']:.4f} | "
        f"{with_elo['accuracy']:.4f} | "
        f"{with_elo['accuracy'] - without_elo['accuracy']:+.4f} | "
        f"{without_elo['balanced_accuracy']:.4f} | "
        f"{with_elo['balanced_accuracy']:.4f} | "
        f"{without_elo['macro_f1']:.4f} | "
        f"{with_elo['macro_f1']:.4f} | "
        f"{without_elo.get('log_loss', float('nan')):.4f} | "
        f"{with_elo.get('log_loss', float('nan')):.4f} |"
    )


def build_report() -> str:
    splits = load_fixture_level_splits()
    with_elo_features = infer_fixture_feature_columns(splits["train"])
    validate_fixture_level_splits(splits, with_elo_features)
    without_elo_features = [
        feature for feature in with_elo_features if feature not in ELO_COLUMNS
    ]

    missing_elo = sorted(set(ELO_COLUMNS).difference(with_elo_features))
    if missing_elo:
        raise ValueError(f"Expected Elo columns missing from fixture features: {missing_elo}")

    without_elo = evaluate_feature_set(splits, without_elo_features)
    with_elo = evaluate_feature_set(splits, with_elo_features)

    lines = [
        "Elo Classical ML Comparison",
        "=" * 40,
        "Dataset: data/processed/fixture_level",
        f"Feature count without Elo: {len(without_elo_features)}",
        f"Feature count with Elo: {len(with_elo_features)}",
        f"Elo columns: {', '.join(ELO_COLUMNS)}",
        "",
        "| Model | Acc no Elo | Acc Elo | Acc diff | Bal Acc no Elo | Bal Acc Elo | Macro F1 no Elo | Macro F1 Elo | Log loss no Elo | Log loss Elo |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name in MODEL_NAMES:
        lines.append(metric_row(model_name, without_elo[model_name], with_elo[model_name]))

    return "\n".join(lines)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = build_report()
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved Elo ML comparison: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
