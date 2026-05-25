from pathlib import Path

import pandas as pd


SPLIT_PATHS = {
    "train": Path("data/raw/splits/train.csv"),
    "val": Path("data/raw/splits/val.csv"),
    "test": Path("data/raw/splits/test.csv"),
}
REPORT_PATH = Path("outputs/reports/feature_report.txt")
TARGET_COL = "FTR"
DATE_COL = "Date"

IDENTIFIER_COLUMNS = {
    "Date",
    "HomeTeam",
    "AwayTeam",
    "Referee",
    "season",
}

KNOWN_LEAKAGE_COLUMNS = {
    TARGET_COL,
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

LEAKAGE_KEYWORDS = (
    "goal",
    "score",
    "result",
    "ftr",
    "fthg",
    "ftag",
    "hthg",
    "htag",
    "htr",
    "shot",
    "corner",
    "foul",
    "card",
    "yellow",
    "red",
)


def load_splits() -> dict[str, pd.DataFrame]:
    missing_paths = [str(path) for path in SPLIT_PATHS.values() if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Missing split files: {', '.join(missing_paths)}")

    return {
        split_name: pd.read_csv(path, parse_dates=[DATE_COL])
        for split_name, path in SPLIT_PATHS.items()
    }


def detect_leakage_columns(columns: list[str]) -> list[str]:
    leakage = set()

    for col in columns:
        col_lower = col.lower()
        if col in KNOWN_LEAKAGE_COLUMNS:
            leakage.add(col)
        elif any(keyword in col_lower for keyword in LEAKAGE_KEYWORDS):
            leakage.add(col)

    return sorted(leakage)


def classify_features(df: pd.DataFrame) -> dict[str, list[str]]:
    columns = list(df.columns)
    leakage_columns = detect_leakage_columns(columns)
    identifier_columns = sorted(col for col in columns if col in IDENTIFIER_COLUMNS)
    excluded = set(leakage_columns) | set(identifier_columns)

    candidate_features = [col for col in columns if col not in excluded]
    numeric_features = sorted(
        col for col in candidate_features if pd.api.types.is_numeric_dtype(df[col])
    )
    categorical_features = sorted(
        col
        for col in candidate_features
        if pd.api.types.is_object_dtype(df[col])
        or pd.api.types.is_categorical_dtype(df[col])
    )

    return {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "leakage_columns": leakage_columns,
        "identifier_columns": identifier_columns,
    }


def format_series(title: str, series: pd.Series) -> str:
    return f"{title}\n{series.to_string()}\n"


def build_report(splits: dict[str, pd.DataFrame]) -> str:
    train_df = splits["train"]
    feature_groups = classify_features(train_df)
    date_ranges = {
        split_name: (df[DATE_COL].min(), df[DATE_COL].max())
        for split_name, df in splits.items()
    }
    chronological_ok = (
        date_ranges["train"][1] < date_ranges["val"][0]
        and date_ranges["val"][1] < date_ranges["test"][0]
    )
    lines = ["Football Prediction Split Inspection", "=" * 44, ""]

    for split_name, df in splits.items():
        lines.extend(
            [
                f"[{split_name}]",
                f"Shape: {df.shape[0]} rows x {df.shape[1]} columns",
                (
                    "Date range: "
                    f"{date_ranges[split_name][0].date()} to "
                    f"{date_ranges[split_name][1].date()}"
                ),
                "",
                "Columns:",
                ", ".join(df.columns),
                "",
                format_series("Dtypes:", df.dtypes.astype(str)),
                format_series("Missing values:", df.isna().sum()),
                format_series(
                    f"Target distribution ({TARGET_COL}):",
                    df[TARGET_COL].value_counts(dropna=False).sort_index(),
                ),
                "",
            ]
        )

    lines.extend(
        [
            "Chronological Split Check",
            "=" * 44,
            f"Status: {'PASS' if chronological_ok else 'FAIL'}",
            (
                "Expected ordering: "
                "train max date < validation min date < "
                "validation max date < test min date"
            ),
            "",
            "Feature Groups Based On Training Columns",
            "=" * 44,
            f"Numeric features ({len(feature_groups['numeric_features'])}):",
            ", ".join(feature_groups["numeric_features"]) or "(none)",
            "",
            f"Categorical features ({len(feature_groups['categorical_features'])}):",
            ", ".join(feature_groups["categorical_features"]) or "(none)",
            "",
            f"Leakage columns ({len(feature_groups['leakage_columns'])}):",
            ", ".join(feature_groups["leakage_columns"]) or "(none)",
            "",
            f"Identifier columns ({len(feature_groups['identifier_columns'])}):",
            ", ".join(feature_groups["identifier_columns"]) or "(none)",
            "",
            "Leakage Note",
            "=" * 44,
            (
                "Leakage columns include current-match outcomes and in-match events. "
                "They must be excluded from pre-match models because they are only known "
                "after kickoff or after the match is finished."
            ),
            "",
        ]
    )

    return "\n".join(lines)


def print_console_summary(splits: dict[str, pd.DataFrame], report: str) -> None:
    print(report)
    print(f"Saved feature summary report to: {REPORT_PATH}")


def main() -> None:
    splits = load_splits()
    report = build_report(splits)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print_console_summary(splits, report)


if __name__ == "__main__":
    main()
