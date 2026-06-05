from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.features.build_team_features import (
    LEAKAGE_COLUMNS_EXCLUDED,
    MIN_PREVIOUS_MATCHES,
    PROJECT_ROOT,
    RAW_DIR,
    WINDOWS,
    add_rolling_features,
    load_raw_matches,
    parse_match_dates,
    require_raw_split_files,
    to_team_centric,
)


PROCESSED_FIXTURE_PATH = PROJECT_ROOT / "data/processed/fixture_level/features.csv"
PROCESSED_FIXTURE_SPLIT_DIR = PROJECT_ROOT / "data/processed/fixture_level"
REPORT_PATH = PROJECT_ROOT / "outputs/reports/fixture_level_feature_report.txt"

TARGET_MAP = {"H": 0, "D": 1, "A": 2}
TARGET_NAMES = ["HomeWin", "Draw", "AwayWin"]


@dataclass(frozen=True)
class FixtureBuildResult:
    raw_matches: pd.DataFrame
    processed: pd.DataFrame
    splits: dict[str, pd.DataFrame]
    feature_columns: list[str]


def rolling_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if any(col.endswith(f"_last_{window}") for window in WINDOWS)
    ]


def prefixed_team_history(
    rolled_team_rows: pd.DataFrame,
    side: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    base_columns = ["match_id", "team", "previous_matches", "rest_days", *feature_columns]
    history = rolled_team_rows[base_columns].copy()
    return history.rename(
        columns={
            "team": f"{side}_team",
            "previous_matches": f"{side}_previous_matches",
            "rest_days": f"{side}_rest_days",
            **{col: f"{side}_{col}" for col in feature_columns},
        }
    )


def add_difference_features(
    fixtures: pd.DataFrame,
    rolling_columns: list[str],
) -> pd.DataFrame:
    df = fixtures.copy()
    numeric_pairs = ["rest_days", *rolling_columns]
    for col in numeric_pairs:
        home_col = f"home_{col}"
        away_col = f"away_{col}"
        if home_col in df.columns and away_col in df.columns:
            df[f"diff_{col}"] = df[home_col] - df[away_col]
    return df


def split_fixture_processed(processed: pd.DataFrame) -> dict[str, pd.DataFrame]:
    raw_split_paths = require_raw_split_files(RAW_DIR)
    raw_train = pd.read_csv(raw_split_paths["train"])
    raw_val = pd.read_csv(raw_split_paths["val"])
    raw_train["Date"] = parse_match_dates(raw_train["Date"])
    raw_val["Date"] = parse_match_dates(raw_val["Date"])

    train_end = raw_train["Date"].max()
    val_end = raw_val["Date"].max()

    train = processed[processed["Date"] <= train_end].copy()
    val = processed[
        (processed["Date"] > train_end) & (processed["Date"] <= val_end)
    ].copy()
    test = processed[processed["Date"] > val_end].copy()
    return {"train": train, "val": val, "test": test}


def fixture_feature_columns(processed: pd.DataFrame) -> list[str]:
    excluded = {
        "match_id",
        "Date",
        "target",
        "target_label",
        "home_previous_matches",
        "away_previous_matches",
    }
    return [col for col in processed.columns if col not in excluded]


def build_fixture_features() -> FixtureBuildResult:
    raw_matches = load_raw_matches()
    team_rows = to_team_centric(raw_matches)
    rolled = add_rolling_features(team_rows)
    rolling_columns = rolling_feature_columns(rolled)

    home_history = prefixed_team_history(rolled, "home", rolling_columns)
    away_history = prefixed_team_history(rolled, "away", rolling_columns)

    fixtures = raw_matches[
        ["match_id", "Date", "season", "HomeTeam", "AwayTeam", "FTR"]
    ].copy()
    fixtures = fixtures.rename(
        columns={"HomeTeam": "home_team", "AwayTeam": "away_team"}
    )
    fixtures["target"] = fixtures["FTR"].map(TARGET_MAP)
    fixtures["target_label"] = fixtures["target"].map(dict(enumerate(TARGET_NAMES)))
    fixtures["month"] = fixtures["Date"].dt.month

    fixtures = fixtures.merge(
        home_history,
        on=["match_id", "home_team"],
        how="left",
    )
    fixtures = fixtures.merge(
        away_history,
        on=["match_id", "away_team"],
        how="left",
    )

    processed = fixtures[
        (fixtures["home_previous_matches"] >= MIN_PREVIOUS_MATCHES)
        & (fixtures["away_previous_matches"] >= MIN_PREVIOUS_MATCHES)
    ].copy()
    processed = processed.drop(columns=["FTR"])
    processed = add_difference_features(processed, rolling_columns)
    processed = processed.sort_values(["Date", "match_id"]).reset_index(drop=True)

    feature_columns = fixture_feature_columns(processed)
    splits = split_fixture_processed(processed)
    return FixtureBuildResult(
        raw_matches=raw_matches,
        processed=processed,
        splits=splits,
        feature_columns=feature_columns,
    )


def target_distribution(df: pd.DataFrame) -> str:
    distribution = (
        df["target"]
        .value_counts(dropna=False)
        .sort_index()
        .rename(index={0: "0 HomeWin", 1: "1 Draw", 2: "2 AwayWin"})
    )
    return distribution.to_string()


def build_report(result: FixtureBuildResult) -> str:
    processed = result.processed
    lines = [
        "Fixture-Level Feature Engineering Report",
        "=" * 48,
        f"Total raw matches: {len(result.raw_matches)}",
        f"Fixture rows after dropping insufficient history: {len(processed)}",
        f"Date range: {processed['Date'].min().date()} to {processed['Date'].max().date()}",
        "Target: 0=HomeWin, 1=Draw, 2=AwayWin",
        "",
        "Split Sizes",
        "=" * 48,
    ]

    for split_name, split_df in result.splits.items():
        lines.append(
            f"{split_name}: {len(split_df)} rows, "
            f"{split_df['Date'].min().date()} to {split_df['Date'].max().date()}"
        )

    lines.extend(["", "Target Distribution Per Split", "=" * 48])
    for split_name, split_df in result.splits.items():
        lines.extend([f"[{split_name}]", target_distribution(split_df), ""])

    lines.extend(
        [
            "Feature Columns",
            "=" * 48,
            f"Total: {len(result.feature_columns)}",
            "\n".join(result.feature_columns),
            "",
            "Leakage Columns Excluded",
            "=" * 48,
            "\n".join(LEAKAGE_COLUMNS_EXCLUDED),
            "",
        ]
    )
    return "\n".join(lines)


def save_outputs(result: FixtureBuildResult) -> None:
    PROCESSED_FIXTURE_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    result.processed.to_csv(PROCESSED_FIXTURE_PATH, index=False)
    for split_name, split_df in result.splits.items():
        split_df.to_csv(PROCESSED_FIXTURE_SPLIT_DIR / f"{split_name}.csv", index=False)

    REPORT_PATH.write_text(build_report(result), encoding="utf-8")


def main() -> None:
    result = build_fixture_features()
    save_outputs(result)
    print(f"Saved fixture-level dataset: {PROCESSED_FIXTURE_PATH}")
    print(f"Saved fixture-level splits: {PROCESSED_FIXTURE_SPLIT_DIR}")
    print(f"Saved feature report: {REPORT_PATH}")
    print(f"Rows after history filter: {len(result.processed)}")


if __name__ == "__main__":
    main()
