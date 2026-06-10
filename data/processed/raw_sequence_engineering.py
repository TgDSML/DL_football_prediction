from __future__ import annotations

from pathlib import Path
import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "premier-league"
SPLIT_DIR = PROJECT_ROOT / "data" / "raw" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "sequences_leakage_safe"

TARGET_MAP = {"H": 0, "D": 1, "A": 2}
TARGET_NAMES = ["HomeWin", "Draw", "AwayWin"]
WINDOWS = (3, 5, 10)

RAW_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR", "Referee",
]

BASE_SEQUENCE_FEATURES = [
    "was_home", "points", "goals_for", "goals_against", "goal_diff",
    "shots_for", "shots_against", "shots_on_target_for", "shots_on_target_against",
    "corners_for", "corners_against", "yellow_cards_for", "yellow_cards_against",
    "red_cards_for", "red_cards_against", "rest_days", "team_elo",
]

ROLLING_FEATURES = [
    feat
    for window in WINDOWS
    for feat in [
        f"points_avg_last_{window}", f"points_sum_last_{window}",
        f"wins_sum_last_{window}", f"draws_sum_last_{window}", f"losses_sum_last_{window}",
        f"goals_for_avg_last_{window}", f"goals_against_avg_last_{window}", f"goal_diff_avg_last_{window}",
        f"shots_for_avg_last_{window}", f"shots_against_avg_last_{window}",
        f"shots_on_target_for_avg_last_{window}", f"shots_on_target_against_avg_last_{window}",
        f"corners_for_avg_last_{window}", f"corners_against_avg_last_{window}",
        f"yellow_cards_for_avg_last_{window}", f"yellow_cards_against_avg_last_{window}",
        f"red_cards_for_avg_last_{window}", f"red_cards_against_avg_last_{window}",
        f"clean_sheets_sum_last_{window}", f"failed_to_score_sum_last_{window}",
    ]
]

AUGMENTED_SEQUENCE_FEATURES = [
    *BASE_SEQUENCE_FEATURES,
    *ROLLING_FEATURES,
    "recent_form_points_weighted",
    "unbeaten_streak",
    "win_streak",
    "losing_streak",
]


@dataclass(frozen=True)
class SequenceSplit:
    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame


@dataclass(frozen=True)
class SequenceBuildResult:
    feature_names: list[str]
    sequence_length: int
    variant: str
    leakage_check_passed: bool
    splits: dict[str, SequenceSplit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe fixture sequences from raw football match CSVs."
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[3, 5, 10])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["home_only", "home_away"],
        default=["home_only", "home_away"],
    )
    parser.add_argument(
        "--feature-mode",
        choices=["raw", "raw_plus_rolling"],
        default="raw_plus_rolling",
    )
    return parser.parse_args()


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


def season_code_from_path(path: Path) -> str:
    return path.stem.replace("season-", "")


def require_raw_split_files(split_dir: Path) -> dict[str, Path]:
    paths = {name: split_dir / f"{name}.csv" for name in ("train", "val", "test")}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw split CSV files: {', '.join(missing)}")
    return paths


def load_raw_matches(raw_dir: Path) -> pd.DataFrame:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    paths = sorted(
        path for path in raw_dir.glob("season-*.csv")
        if path.is_file()
    )

    if not paths:
        raise FileNotFoundError(f"No season-*.csv files found in {raw_dir}")

    frames = []
    for path in paths:
        df = pd.read_csv(path)

        available_columns = [col for col in RAW_COLUMNS if col in df.columns]
        if not available_columns:
            continue

        df = df[available_columns].copy()

        if "Referee" not in df.columns:
            df["Referee"] = pd.NA

        if "season" not in df.columns:
            df["season"] = season_code_from_path(path)

        df["season"] = df["season"].astype(str)
        df["source_file"] = path.name
        frames.append(df)

    if not frames:
        raise ValueError(
            f"Found CSV files in {raw_dir}, but none had the expected football columns."
        )

    matches = pd.concat(frames, ignore_index=True)
    matches["Date"] = parse_match_dates(matches["Date"])
    matches = matches.sort_values(["Date", "season", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    matches["match_id"] = matches.index
    return matches


def compute_elo_ratings(
    matches: pd.DataFrame,
    base_rating: float = 1500.0,
    k_factor: float = 20.0,
) -> pd.DataFrame:
    ratings: dict[str, float] = {}
    rows = []

    for row in matches.itertuples(index=False):
        home = row.HomeTeam
        away = row.AwayTeam

        home_pre = ratings.get(home, base_rating)
        away_pre = ratings.get(away, base_rating)

        expected_home = 1.0 / (1.0 + 10 ** ((away_pre - home_pre) / 400.0))
        expected_away = 1.0 - expected_home

        if row.FTR == "H":
            actual_home, actual_away = 1.0, 0.0
        elif row.FTR == "D":
            actual_home, actual_away = 0.5, 0.5
        else:
            actual_home, actual_away = 0.0, 1.0

        home_post = home_pre + k_factor * (actual_home - expected_home)
        away_post = away_pre + k_factor * (actual_away - expected_away)

        ratings[home] = home_post
        ratings[away] = away_post

        rows.append(
            {
                "match_id": int(row.match_id),
                "HomeTeam": home,
                "AwayTeam": away,
                "home_elo": float(home_pre),
                "away_elo": float(away_pre),
            }
        )

    return pd.DataFrame(rows)


def team_elo_features(elo_df: pd.DataFrame) -> pd.DataFrame:
    home = elo_df[["match_id", "HomeTeam", "AwayTeam", "home_elo"]].rename(
        columns={"HomeTeam": "team", "AwayTeam": "opponent", "home_elo": "team_elo"}
    )
    away = elo_df[["match_id", "AwayTeam", "HomeTeam", "away_elo"]].rename(
        columns={"AwayTeam": "team", "HomeTeam": "opponent", "away_elo": "team_elo"}
    )
    return pd.concat([home, away], ignore_index=True)


def _team_points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def build_sequence_team_rows(matches: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "Date": matches["Date"],
            "season": matches["season"],
            "team": matches["HomeTeam"],
            "opponent": matches["AwayTeam"],
            "was_home": 1.0,
            "goals_for": matches["FTHG"],
            "goals_against": matches["FTAG"],
            "shots_for": matches["HS"],
            "shots_against": matches["AS"],
            "shots_on_target_for": matches["HST"],
            "shots_on_target_against": matches["AST"],
            "corners_for": matches["HC"],
            "corners_against": matches["AC"],
            "yellow_cards_for": matches["HY"],
            "yellow_cards_against": matches["AY"],
            "red_cards_for": matches["HR"],
            "red_cards_against": matches["AR"],
        }
    )

    away = pd.DataFrame(
        {
            "match_id": matches["match_id"],
            "Date": matches["Date"],
            "season": matches["season"],
            "team": matches["AwayTeam"],
            "opponent": matches["HomeTeam"],
            "was_home": 0.0,
            "goals_for": matches["FTAG"],
            "goals_against": matches["FTHG"],
            "shots_for": matches["AS"],
            "shots_against": matches["HS"],
            "shots_on_target_for": matches["AST"],
            "shots_on_target_against": matches["HST"],
            "corners_for": matches["AC"],
            "corners_against": matches["HC"],
            "yellow_cards_for": matches["AY"],
            "yellow_cards_against": matches["HY"],
            "red_cards_for": matches["AR"],
            "red_cards_against": matches["HR"],
        }
    )

    team_rows = pd.concat([home, away], ignore_index=True)
    team_rows["points"] = team_rows.apply(
        lambda r: _team_points(int(r["goals_for"]), int(r["goals_against"])),
        axis=1,
    )
    team_rows["goal_diff"] = team_rows["goals_for"] - team_rows["goals_against"]
    team_rows["wins"] = (team_rows["points"] == 3).astype(float)
    team_rows["draws"] = (team_rows["points"] == 1).astype(float)
    team_rows["losses"] = (team_rows["points"] == 0).astype(float)
    team_rows["clean_sheets"] = (team_rows["goals_against"] == 0).astype(float)
    team_rows["failed_to_score"] = (team_rows["goals_for"] == 0).astype(float)
    team_rows = team_rows.sort_values(["team", "Date", "match_id"]).reset_index(drop=True)
    team_rows["rest_days"] = team_rows.groupby("team")["Date"].diff().dt.days.fillna(0.0)
    return team_rows


def _weighted_recent_points(prior_points: pd.Series) -> float:
    values = prior_points.tail(5).to_numpy(dtype=np.float32, copy=True)
    if len(values) == 0:
        return 0.0
    weights = np.arange(1, len(values) + 1, dtype=np.float32)
    return float(np.average(values, weights=weights))


def _prior_streaks(history: pd.DataFrame) -> pd.DataFrame:
    rows = []
    unbeaten_streak = 0
    win_streak = 0
    losing_streak = 0

    for row in history.itertuples(index=False):
        rows.append(
            {
                "match_id": int(row.match_id),
                "team": row.team,
                "unbeaten_streak": float(unbeaten_streak),
                "win_streak": float(win_streak),
                "losing_streak": float(losing_streak),
            }
        )

        points = int(row.points)
        unbeaten_streak = unbeaten_streak + 1 if points > 0 else 0
        win_streak = win_streak + 1 if points == 3 else 0
        losing_streak = losing_streak + 1 if points == 0 else 0

    return pd.DataFrame(rows)


def add_shifted_rolling_sequence_features(team_rows: pd.DataFrame) -> pd.DataFrame:
    df = team_rows.copy()
    grouped = df.groupby("team", group_keys=False)

    rolling_sources = {
        "points": "points",
        "wins": "wins",
        "draws": "draws",
        "losses": "losses",
        "goals_for": "goals_for",
        "goals_against": "goals_against",
        "goal_diff": "goal_diff",
        "shots_for": "shots_for",
        "shots_against": "shots_against",
        "shots_on_target_for": "shots_on_target_for",
        "shots_on_target_against": "shots_on_target_against",
        "corners_for": "corners_for",
        "corners_against": "corners_against",
        "yellow_cards_for": "yellow_cards_for",
        "yellow_cards_against": "yellow_cards_against",
        "red_cards_for": "red_cards_for",
        "red_cards_against": "red_cards_against",
        "clean_sheets": "clean_sheets",
        "failed_to_score": "failed_to_score",
    }

    sum_metrics = {"points", "wins", "draws", "losses", "clean_sheets", "failed_to_score"}

    for metric_name, source_col in rolling_sources.items():
        shifted = grouped[source_col].shift(1)

        for window in WINDOWS:
            rolling = shifted.groupby(df["team"]).rolling(window=window, min_periods=1)

            if metric_name in sum_metrics:
                df[f"{metric_name}_sum_last_{window}"] = rolling.sum().reset_index(level=0, drop=True)

            if metric_name == "points" or metric_name not in sum_metrics:
                df[f"{metric_name}_avg_last_{window}"] = rolling.mean().reset_index(level=0, drop=True)

    weighted_form = []
    streak_frames = []

    for _, history in df.groupby("team", sort=False):
        history = history.sort_values(["Date", "match_id"])
        weighted_form.extend(
            _weighted_recent_points(history["points"].iloc[:i])
            for i in range(len(history))
        )
        streak_frames.append(_prior_streaks(history))

    df["recent_form_points_weighted"] = weighted_form
    streaks = pd.concat(streak_frames, ignore_index=True)
    df = df.merge(streaks, on=["match_id", "team"], how="left")
    df[AUGMENTED_SEQUENCE_FEATURES] = df[AUGMENTED_SEQUENCE_FEATURES].fillna(0.0)
    return df


def sequence_features_for_mode(feature_mode: str) -> list[str]:
    if feature_mode == "raw":
        return BASE_SEQUENCE_FEATURES
    if feature_mode == "raw_plus_rolling":
        return AUGMENTED_SEQUENCE_FEATURES
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def prepare_team_rows_for_feature_mode(matches: pd.DataFrame, feature_mode: str) -> pd.DataFrame:
    team_rows = build_sequence_team_rows(matches)
    elo_features = team_elo_features(compute_elo_ratings(matches))
    team_rows = team_rows.merge(
        elo_features[["match_id", "team", "opponent", "team_elo"]],
        on=["match_id", "team", "opponent"],
        how="left",
    )
    if feature_mode == "raw":
        return team_rows
    return add_shifted_rolling_sequence_features(team_rows)


def split_bounds(split_dir: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
    raw_split_paths = require_raw_split_files(split_dir)

    raw_train = pd.read_csv(raw_split_paths["train"])
    raw_val = pd.read_csv(raw_split_paths["val"])

    raw_train["Date"] = parse_match_dates(raw_train["Date"])
    raw_val["Date"] = parse_match_dates(raw_val["Date"])

    return raw_train["Date"].max(), raw_val["Date"].max()


def split_name_for_date(target_date: pd.Timestamp, train_end: pd.Timestamp, val_end: pd.Timestamp) -> str:
    if target_date <= train_end:
        return "train"
    if target_date <= val_end:
        return "val"
    return "test"


def build_team_history_index(team_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        team: hist.sort_values(["Date", "match_id"]).reset_index(drop=True)
        for team, hist in team_rows.groupby("team", sort=False)
    }


def strict_prior_history(
    history: pd.DataFrame,
    target_date: pd.Timestamp,
    sequence_length: int,
) -> pd.DataFrame | None:
    prior_matches = history[history["Date"] < target_date]
    if len(prior_matches) < sequence_length:
        return None
    return prior_matches.tail(sequence_length).copy()


def build_sequences(
    matches: pd.DataFrame,
    split_dir: Path,
    sequence_length: int,
    feature_mode: str,
    variant: str,
) -> SequenceBuildResult:
    matches = matches.sort_values(["Date", "season", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    feature_names_base = sequence_features_for_mode(feature_mode)
    team_rows = prepare_team_rows_for_feature_mode(matches, feature_mode)
    histories = build_team_history_index(team_rows)
    train_end, val_end = split_bounds(split_dir)

    split_arrays = {"train": [], "val": [], "test": []}
    split_labels = {"train": [], "val": [], "test": []}
    metadata_rows = {"train": [], "val": [], "test": []}
    leakage_check_passed = True

    if variant == "home_only":
        feature_names = feature_names_base
    else:
        feature_names = [
            *[f"home_{f}" for f in feature_names_base],
            *[f"away_{f}" for f in feature_names_base],
        ]

    for fixture in matches.itertuples(index=False):
        split_name = split_name_for_date(fixture.Date, train_end, val_end)
        home_window = strict_prior_history(histories[fixture.HomeTeam], fixture.Date, sequence_length)

        if variant == "home_only":
            if home_window is None:
                continue

            history_end_date = home_window["Date"].max()
            if not bool(history_end_date < fixture.Date):
                leakage_check_passed = False
                raise ValueError("Leakage detected: home sequence contains current/future match")

            sequence = home_window[feature_names_base].to_numpy(dtype=np.float32, copy=True)
            meta = {
                "match_id": int(fixture.match_id),
                "Date": fixture.Date,
                "HomeTeam": fixture.HomeTeam,
                "AwayTeam": fixture.AwayTeam,
                "FTR": fixture.FTR,
                "target": int(TARGET_MAP[fixture.FTR]),
                "target_label": TARGET_NAMES[TARGET_MAP[fixture.FTR]],
                "history_start_date": home_window["Date"].min(),
                "history_end_date": history_end_date,
                "split": split_name,
            }
        else:
            away_window = strict_prior_history(histories[fixture.AwayTeam], fixture.Date, sequence_length)
            if home_window is None or away_window is None:
                continue

            home_end = home_window["Date"].max()
            away_end = away_window["Date"].max()

            if not bool(home_end < fixture.Date and away_end < fixture.Date):
                leakage_check_passed = False
                raise ValueError("Leakage detected: home/away sequence contains current/future match")

            sequence = np.concatenate(
                [
                    home_window[feature_names_base].to_numpy(dtype=np.float32, copy=True),
                    away_window[feature_names_base].to_numpy(dtype=np.float32, copy=True),
                ],
                axis=1,
            )

            meta = {
                "match_id": int(fixture.match_id),
                "Date": fixture.Date,
                "HomeTeam": fixture.HomeTeam,
                "AwayTeam": fixture.AwayTeam,
                "FTR": fixture.FTR,
                "target": int(TARGET_MAP[fixture.FTR]),
                "target_label": TARGET_NAMES[TARGET_MAP[fixture.FTR]],
                "home_history_start_date": home_window["Date"].min(),
                "home_history_end_date": home_end,
                "away_history_start_date": away_window["Date"].min(),
                "away_history_end_date": away_end,
                "split": split_name,
            }

        split_arrays[split_name].append(sequence)
        split_labels[split_name].append(int(TARGET_MAP[fixture.FTR]))
        metadata_rows[split_name].append(meta)

    splits = {}
    num_features = len(feature_names)

    for split_name in ("train", "val", "test"):
        metadata = (
            pd.DataFrame(metadata_rows[split_name]).sort_values("Date").reset_index(drop=True)
            if metadata_rows[split_name]
            else pd.DataFrame()
        )

        if split_arrays[split_name]:
            X = np.stack(split_arrays[split_name]).astype(np.float32, copy=False)
            y = np.asarray(split_labels[split_name], dtype=np.int64)
        else:
            X = np.empty((0, sequence_length, num_features), dtype=np.float32)
            y = np.empty((0,), dtype=np.int64)

        splits[split_name] = SequenceSplit(X=X, y=y, metadata=metadata)

    return SequenceBuildResult(
        feature_names=feature_names,
        sequence_length=sequence_length,
        variant=variant,
        leakage_check_passed=leakage_check_passed,
        splits=splits,
    )


def save_result(result: SequenceBuildResult, output_dir: Path) -> dict[str, tuple[Path, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = result.variant if result.sequence_length == 5 else f"{result.variant}_seq{result.sequence_length}"

    feature_path = output_dir / f"{prefix}_feature_names.json"
    feature_path.write_text(json.dumps(result.feature_names, indent=2), encoding="utf-8")

    saved = {}
    for split_name, split in result.splits.items():
        npz_path = output_dir / f"{prefix}_{split_name}.npz"
        metadata_path = output_dir / f"{prefix}_{split_name}_metadata.csv"

        np.savez(
            npz_path,
            X=split.X,
            y=split.y,
            feature_names=np.asarray(result.feature_names),
        )
        split.metadata.to_csv(metadata_path, index=False)
        saved[split_name] = (npz_path, metadata_path)

    return saved


def build_summary(result: SequenceBuildResult, saved_paths: dict[str, tuple[Path, Path]]) -> str:
    lines = [
        f"Leakage-Safe Sequence Build Summary ({result.variant})",
        "=" * 44,
        f"Sequence length: {result.sequence_length}",
        f"Features per timestep: {len(result.feature_names)}",
        f"Leakage check: {'PASSED' if result.leakage_check_passed else 'FAILED'}",
        "",
    ]

    for split_name in ("train", "val", "test"):
        split = result.splits[split_name]
        lines.append(f"[{split_name}]")
        lines.append(f"X shape: {split.X.shape}")
        lines.append(f"y shape: {split.y.shape}")

        if not split.metadata.empty:
            lines.append(
                f"Date range: {split.metadata['Date'].min().date()} to {split.metadata['Date'].max().date()}"
            )
        else:
            lines.append("Date range: n/a")

        npz_path, metadata_path = saved_paths[split_name]
        lines.append(f"Saved npz: {npz_path}")
        lines.append(f"Saved metadata: {metadata_path}")
        lines.append("")

    return "\n".join(lines).rstrip()


def main() -> None:
    args = parse_args()

    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("raw_dir =", args.raw_dir)
    print("split_dir =", args.split_dir)
    print("output_dir =", args.output_dir)

    matches = load_raw_matches(args.raw_dir)

    all_summaries = []
    for variant in args.variants:
        for seq_len in args.sequence_lengths:
            result = build_sequences(
                matches=matches,
                split_dir=args.split_dir,
                sequence_length=seq_len,
                feature_mode=args.feature_mode,
                variant=variant,
            )
            saved = save_result(result, args.output_dir)
            all_summaries.append(build_summary(result, saved))

    report_path = args.output_dir / "build_report.txt"
    report_path.write_text("\n\n".join(all_summaries), encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()