from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.build_team_features import (
    CURRENT_MATCH_COLUMNS,
    LEAKAGE_COLUMNS_EXCLUDED,
    PROCESSED_SPLIT_DIR,
)
from src.features.build_fixture_features import PROCESSED_FIXTURE_SPLIT_DIR


TARGET_COLUMN = "target"
TARGET_NAMES = ["win", "draw", "loss"]
FIXTURE_TARGET_NAMES = ["HomeWin", "Draw", "AwayWin"]

METADATA_COLUMNS = {
    "match_id",
    "Date",
    "target",
    "previous_matches",
    "opponent_previous_matches",
}
KNOWN_CATEGORICAL_FEATURES = {"season", "team", "opponent"}
FIXTURE_METADATA_COLUMNS = {
    "match_id",
    "Date",
    "target",
    "target_label",
    "home_previous_matches",
    "away_previous_matches",
}
FIXTURE_CATEGORICAL_FEATURES = {"season", "home_team", "away_team"}
FORBIDDEN_DIRECT_INPUTS = set(LEAKAGE_COLUMNS_EXCLUDED) | set(CURRENT_MATCH_COLUMNS)


def load_processed_splits(
    split_dir: Path = PROCESSED_SPLIT_DIR,
) -> dict[str, pd.DataFrame]:
    splits = {}
    for split_name in ("train", "val", "test"):
        path = split_dir / f"{split_name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `python scripts/build_features.py` first."
            )
        splits[split_name] = pd.read_csv(path, parse_dates=["Date"])
    return splits


def load_fixture_level_splits(
    split_dir: Path = PROCESSED_FIXTURE_SPLIT_DIR,
) -> dict[str, pd.DataFrame]:
    splits = {}
    for split_name in ("train", "val", "test"):
        path = split_dir / f"{split_name}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `python scripts/build_fixture_features.py` first."
            )
        splits[split_name] = pd.read_csv(path, parse_dates=["Date"])
    return splits


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    features = [col for col in df.columns if col not in METADATA_COLUMNS]
    leakage = sorted(FORBIDDEN_DIRECT_INPUTS.intersection(features))
    if leakage:
        raise ValueError(
            "Processed features contain direct current-match leakage columns: "
            + ", ".join(leakage)
        )
    return features


def infer_fixture_feature_columns(df: pd.DataFrame) -> list[str]:
    features = [col for col in df.columns if col not in FIXTURE_METADATA_COLUMNS]
    leakage = sorted(FORBIDDEN_DIRECT_INPUTS.intersection(features))
    if leakage:
        raise ValueError(
            "Fixture-level features contain direct current-match leakage columns: "
            + ", ".join(leakage)
        )
    return features


def split_feature_types(feature_columns: list[str]) -> tuple[list[str], list[str]]:
    categorical = [col for col in feature_columns if col in KNOWN_CATEGORICAL_FEATURES]
    numeric = [col for col in feature_columns if col not in categorical]
    return numeric, categorical


def split_fixture_feature_types(
    feature_columns: list[str],
) -> tuple[list[str], list[str]]:
    categorical = [col for col in feature_columns if col in FIXTURE_CATEGORICAL_FEATURES]
    numeric = [col for col in feature_columns if col not in categorical]
    return numeric, categorical


def validate_processed_splits(
    splits: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> None:
    required = set(feature_columns) | {TARGET_COLUMN, "Date"}
    for split_name, df in splits.items():
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"{split_name} split is missing columns: {missing}")

        leakage = sorted(FORBIDDEN_DIRECT_INPUTS.intersection(feature_columns))
        if leakage:
            raise ValueError(
                f"{split_name} split uses leakage columns as features: {leakage}"
            )

        null_targets = int(df[TARGET_COLUMN].isna().sum())
        if null_targets:
            raise ValueError(f"{split_name} split has {null_targets} missing targets")


def validate_fixture_level_splits(
    splits: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> None:
    required = set(feature_columns) | {TARGET_COLUMN, "Date"}
    previous_max_date = None
    for split_name in ("train", "val", "test"):
        df = splits[split_name]
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"{split_name} split is missing columns: {missing}")

        leakage = sorted(FORBIDDEN_DIRECT_INPUTS.intersection(feature_columns))
        if leakage:
            raise ValueError(
                f"{split_name} fixture split uses leakage columns as features: {leakage}"
            )

        null_targets = int(df[TARGET_COLUMN].isna().sum())
        if null_targets:
            raise ValueError(f"{split_name} split has {null_targets} missing targets")

        min_date = df["Date"].min()
        max_date = df["Date"].max()
        if previous_max_date is not None and min_date <= previous_max_date:
            raise ValueError(
                "Fixture-level splits are not chronological: "
                f"{split_name} starts at {min_date.date()} after previous "
                f"split ended at {previous_max_date.date()}"
            )
        previous_max_date = max_date


def target_distribution(df: pd.DataFrame) -> str:
    distribution = (
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .rename(index={0: "0 win", 1: "1 draw", 2: "2 loss"})
    )
    return distribution.to_string()


def fixture_target_distribution(df: pd.DataFrame) -> str:
    distribution = (
        df[TARGET_COLUMN]
        .value_counts()
        .sort_index()
        .rename(index={0: "0 HomeWin", 1: "1 Draw", 2: "2 AwayWin"})
    )
    return distribution.to_string()
