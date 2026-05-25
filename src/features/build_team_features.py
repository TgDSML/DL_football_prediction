from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed/team_centric_features.csv")
PROCESSED_SPLIT_DIR = Path("data/processed/splits")
REPORT_PATH = Path("outputs/reports/preprocessing_report.txt")

RAW_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
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
    "Referee",
]
WINDOWS = (3, 5, 10)
MIN_PREVIOUS_MATCHES = 5
TARGET_MAP = {"win": 0, "draw": 1, "loss": 2}
RESULT_POINTS = {"win": 3, "draw": 1, "loss": 0}
LEAKAGE_COLUMNS_EXCLUDED = [
    "FTHG",
    "FTAG",
    "FTR",
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
]
CURRENT_MATCH_COLUMNS = [
    "goals_for",
    "goals_against",
    "result_from_team_perspective",
    "points",
    "win",
    "draw",
    "loss",
    "clean_sheet",
    "failed_to_score",
    "goal_diff",
    "shots",
    "shots_on_target",
    "fouls",
    "corners",
    "yellow_cards",
    "red_cards",
]


@dataclass(frozen=True)
class BuildResult:
    raw_matches: pd.DataFrame
    team_rows: pd.DataFrame
    processed: pd.DataFrame
    splits: dict[str, pd.DataFrame]
    feature_columns: list[str]


def season_code_from_path(path: Path) -> str:
    return path.stem.replace("season-", "")


def load_raw_matches(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    paths = sorted(
        path
        for path in raw_dir.glob("season-*.csv")
        if "splits" not in path.parts
    )
    if not paths:
        raise FileNotFoundError(f"No season CSV files found in {raw_dir}")

    frames = []
    for path in paths:
        df = pd.read_csv(path)
        available_columns = [col for col in RAW_COLUMNS if col in df.columns]
        df = df[available_columns].copy()
        if "Referee" not in df.columns:
            df["Referee"] = pd.NA
        df["season"] = season_code_from_path(path)
        df["source_file"] = path.name
        frames.append(df)

    matches = pd.concat(frames, ignore_index=True)
    matches["Date"] = parse_match_dates(matches["Date"])
    matches = matches.sort_values(["Date", "season", "HomeTeam", "AwayTeam"])
    matches = matches.reset_index(drop=True)
    matches["match_id"] = matches.index
    return matches


def parse_match_dates(dates: pd.Series) -> pd.Series:
    date_strings = dates.astype(str)
    iso_mask = date_strings.str.match(r"^\d{4}-\d{2}-\d{2}$")
    parsed = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")

    if iso_mask.any():
        parsed.loc[iso_mask] = pd.to_datetime(
            date_strings.loc[iso_mask],
            format="%Y-%m-%d",
            errors="raise",
        )
    if (~iso_mask).any():
        parsed.loc[~iso_mask] = pd.to_datetime(
            date_strings.loc[~iso_mask],
            dayfirst=True,
            errors="raise",
        )

    return parsed


def team_result(goals_for: int, goals_against: int) -> str:
    if goals_for > goals_against:
        return "win"
    if goals_for == goals_against:
        return "draw"
    return "loss"


def to_team_centric(matches: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "Date": matches["Date"],
            "season": matches["season"],
            "team": matches["HomeTeam"],
            "opponent": matches["AwayTeam"],
            "is_home": 1,
            "goals_for": matches["FTHG"],
            "goals_against": matches["FTAG"],
            "shots": matches["HS"],
            "shots_on_target": matches["HST"],
            "fouls": matches["HF"],
            "corners": matches["HC"],
            "yellow_cards": matches["HY"],
            "red_cards": matches["HR"],
        }
    )
    away = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "Date": matches["Date"],
            "season": matches["season"],
            "team": matches["AwayTeam"],
            "opponent": matches["HomeTeam"],
            "is_home": 0,
            "goals_for": matches["FTAG"],
            "goals_against": matches["FTHG"],
            "shots": matches["AS"],
            "shots_on_target": matches["AST"],
            "fouls": matches["AF"],
            "corners": matches["AC"],
            "yellow_cards": matches["AY"],
            "red_cards": matches["AR"],
        }
    )

    rows = pd.concat([home, away], ignore_index=True)
    rows["result_from_team_perspective"] = rows.apply(
        lambda row: team_result(row["goals_for"], row["goals_against"]),
        axis=1,
    )
    rows["target"] = rows["result_from_team_perspective"].map(TARGET_MAP)
    rows["points"] = rows["result_from_team_perspective"].map(RESULT_POINTS)
    rows["win"] = (rows["result_from_team_perspective"] == "win").astype(int)
    rows["draw"] = (rows["result_from_team_perspective"] == "draw").astype(int)
    rows["loss"] = (rows["result_from_team_perspective"] == "loss").astype(int)
    rows["clean_sheet"] = (rows["goals_against"] == 0).astype(int)
    rows["failed_to_score"] = (rows["goals_for"] == 0).astype(int)
    rows["goal_diff"] = rows["goals_for"] - rows["goals_against"]
    rows = rows.sort_values(["team", "Date", "match_id"]).reset_index(drop=True)
    rows["previous_matches"] = rows.groupby("team").cumcount()
    rows["rest_days"] = rows.groupby("team")["Date"].diff().dt.days
    return rows


def add_rolling_features(team_rows: pd.DataFrame) -> pd.DataFrame:
    df = team_rows.copy()
    base_metrics = {
        "points": "points",
        "wins": "win",
        "draws": "draw",
        "losses": "loss",
        "goals_for": "goals_for",
        "goals_against": "goals_against",
        "goal_diff": "goal_diff",
        "avg_goals_for": "goals_for",
        "avg_goals_against": "goals_against",
        "clean_sheets": "clean_sheet",
        "failed_to_score": "failed_to_score",
        "avg_shots": "shots",
        "avg_shots_on_target": "shots_on_target",
        "avg_corners": "corners",
        "avg_yellow_cards": "yellow_cards",
        "avg_red_cards": "red_cards",
    }
    average_metrics = {
        "avg_goals_for",
        "avg_goals_against",
        "avg_shots",
        "avg_shots_on_target",
        "avg_corners",
        "avg_yellow_cards",
        "avg_red_cards",
    }

    grouped = df.groupby("team", group_keys=False)
    for feature_name, source_col in base_metrics.items():
        shifted = grouped[source_col].shift(1)
        for window in WINDOWS:
            rolling = shifted.groupby(df["team"]).rolling(
                window=window,
                min_periods=1,
            )
            output_col = f"{feature_name}_last_{window}"
            if feature_name in average_metrics:
                df[output_col] = rolling.mean().reset_index(level=0, drop=True)
            else:
                df[output_col] = rolling.sum().reset_index(level=0, drop=True)

    return df


def add_opponent_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    opponent_columns = [
        "match_id",
        "team",
        "previous_matches",
        "rest_days",
        *feature_columns,
    ]
    opponent_df = df[opponent_columns].copy()
    opponent_df = opponent_df.rename(
        columns={
            "team": "opponent",
            "previous_matches": "opponent_previous_matches",
            "rest_days": "opponent_rest_days",
            **{col: f"opponent_{col}" for col in feature_columns},
        }
    )
    return df.merge(opponent_df, on=["match_id", "opponent"], how="left")


def split_processed(processed: pd.DataFrame) -> dict[str, pd.DataFrame]:
    raw_split_dir = RAW_DIR / "splits"
    raw_train = pd.read_csv(raw_split_dir / "train.csv")
    raw_val = pd.read_csv(raw_split_dir / "val.csv")
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


def feature_columns_from(processed: pd.DataFrame) -> list[str]:
    excluded = {
        "match_id",
        "Date",
        "team",
        "opponent",
        "target",
        "previous_matches",
        "opponent_previous_matches",
    }
    return [col for col in processed.columns if col not in excluded]


def build_report(result: BuildResult) -> str:
    processed = result.processed
    lines = [
        "Team-Centric Feature Engineering Report",
        "=" * 46,
        f"Total raw matches: {len(result.raw_matches)}",
        f"Total team-centric rows: {len(result.team_rows)}",
        f"Rows after dropping insufficient history: {len(processed)}",
        f"Date range: {processed['Date'].min().date()} to {processed['Date'].max().date()}",
        "",
        "Split Sizes",
        "=" * 46,
    ]

    for split_name, split_df in result.splits.items():
        lines.append(f"{split_name}: {len(split_df)}")

    lines.extend(["", "Target Distribution Per Split", "=" * 46])
    for split_name, split_df in result.splits.items():
        distribution = (
            split_df["target"]
            .value_counts(dropna=False)
            .sort_index()
            .rename(index={0: "0 win", 1: "1 draw", 2: "2 loss"})
        )
        lines.extend([f"[{split_name}]", distribution.to_string(), ""])

    lines.extend(
        [
            "Feature Columns",
            "=" * 46,
            "\n".join(result.feature_columns),
            "",
            "Leakage Columns Excluded",
            "=" * 46,
            "\n".join(LEAKAGE_COLUMNS_EXCLUDED),
            "",
        ]
    )
    return "\n".join(lines)


def build_team_features() -> BuildResult:
    raw_matches = load_raw_matches()
    team_rows = to_team_centric(raw_matches)
    rolled = add_rolling_features(team_rows)

    rolling_feature_columns = [
        col
        for col in rolled.columns
        if any(col.endswith(f"_last_{window}") for window in WINDOWS)
    ]
    with_opponents = add_opponent_features(rolled, rolling_feature_columns)
    with_opponents["opponent_rest_days"] = with_opponents["opponent_rest_days"]
    with_opponents["match_month"] = with_opponents["Date"].dt.month

    processed_full = with_opponents[
        (with_opponents["previous_matches"] >= MIN_PREVIOUS_MATCHES)
        & (with_opponents["opponent_previous_matches"] >= MIN_PREVIOUS_MATCHES)
    ].copy()
    processed = processed_full.drop(columns=CURRENT_MATCH_COLUMNS)
    processed = processed.sort_values(["Date", "match_id", "is_home"]).reset_index(
        drop=True
    )

    feature_columns = feature_columns_from(processed)
    splits = split_processed(processed)
    return BuildResult(
        raw_matches=raw_matches,
        team_rows=team_rows,
        processed=processed,
        splits=splits,
        feature_columns=feature_columns,
    )


def save_outputs(result: BuildResult) -> None:
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    result.processed.to_csv(PROCESSED_DATA_PATH, index=False)
    for split_name, split_df in result.splits.items():
        split_df.to_csv(PROCESSED_SPLIT_DIR / f"{split_name}.csv", index=False)

    REPORT_PATH.write_text(build_report(result), encoding="utf-8")


def main() -> None:
    result = build_team_features()
    save_outputs(result)
    print(f"Saved processed dataset: {PROCESSED_DATA_PATH}")
    print(f"Saved processed splits: {PROCESSED_SPLIT_DIR}")
    print(f"Saved preprocessing report: {REPORT_PATH}")
    print(f"Rows after history filter: {len(result.processed)}")


if __name__ == "__main__":
    main()
