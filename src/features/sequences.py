from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.elo import compute_elo_ratings, team_elo_features
from src.features.build_team_features import (
    RAW_DIR,
    load_raw_matches,
    parse_match_dates,
    require_raw_split_files,
)


TARGET_MAP = {"H": 0, "D": 1, "A": 2}
TARGET_NAMES = ["HomeWin", "Draw", "AwayWin"]
SEQUENCE_LENGTH = 5
SEQUENCE_OUTPUT_DIR = Path("data/processed/sequences")
HOME_ONLY_PREFIX = "home_only"
HOME_AWAY_PREFIX = "home_away"
RAW_FEATURE_MODE = "raw"
RAW_PLUS_ROLLING_FEATURE_MODE = "raw_plus_rolling"
FEATURE_MODES = (RAW_FEATURE_MODE, RAW_PLUS_ROLLING_FEATURE_MODE)
SEQUENCE_FEATURES = [
    "was_home",
    "points",
    "goals_for",
    "goals_against",
    "goal_diff",
    "shots_for",
    "shots_against",
    "shots_on_target_for",
    "shots_on_target_against",
    "corners_for",
    "corners_against",
    "yellow_cards_for",
    "yellow_cards_against",
    "red_cards_for",
    "red_cards_against",
    "rest_days",
    "team_elo",
]
ROLLING_WINDOWS = (3, 5, 10)
ROLLING_BASE_FEATURES = [
    "points",
    "wins",
    "draws",
    "losses",
    "goals_for",
    "goals_against",
    "goal_diff",
    "shots_for",
    "shots_against",
    "shots_on_target_for",
    "shots_on_target_against",
    "corners_for",
    "corners_against",
    "yellow_cards_for",
    "yellow_cards_against",
    "red_cards_for",
    "red_cards_against",
    "clean_sheets",
    "failed_to_score",
]
ROLLING_FEATURES: list[str] = [
    feature
    for window in ROLLING_WINDOWS
    for feature in [
        f"points_avg_last_{window}",
        f"points_sum_last_{window}",
        f"wins_sum_last_{window}",
        f"draws_sum_last_{window}",
        f"losses_sum_last_{window}",
        f"goals_for_avg_last_{window}",
        f"goals_against_avg_last_{window}",
        f"goal_diff_avg_last_{window}",
        f"shots_for_avg_last_{window}",
        f"shots_against_avg_last_{window}",
        f"shots_on_target_for_avg_last_{window}",
        f"shots_on_target_against_avg_last_{window}",
        f"corners_for_avg_last_{window}",
        f"corners_against_avg_last_{window}",
        f"yellow_cards_for_avg_last_{window}",
        f"yellow_cards_against_avg_last_{window}",
        f"red_cards_for_avg_last_{window}",
        f"red_cards_against_avg_last_{window}",
        f"clean_sheets_sum_last_{window}",
        f"failed_to_score_sum_last_{window}",
    ]
]
AUGMENTED_SEQUENCE_FEATURES = [
    *SEQUENCE_FEATURES,
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
class HomeOnlySequenceBuildResult:
    feature_names: list[str]
    sequence_length: int
    leakage_check_passed: bool
    splits: dict[str, SequenceSplit]


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
        lambda row: _team_points(int(row["goals_for"]), int(row["goals_against"])),
        axis=1,
    )
    team_rows["goal_diff"] = team_rows["goals_for"] - team_rows["goals_against"]
    team_rows["wins"] = (team_rows["points"] == 3).astype(float)
    team_rows["draws"] = (team_rows["points"] == 1).astype(float)
    team_rows["losses"] = (team_rows["points"] == 0).astype(float)
    team_rows["clean_sheets"] = (team_rows["goals_against"] == 0).astype(float)
    team_rows["failed_to_score"] = (team_rows["goals_for"] == 0).astype(float)
    team_rows = team_rows.sort_values(["team", "Date", "match_id"]).reset_index(drop=True)
    team_rows["rest_days"] = team_rows.groupby("team")["Date"].diff().dt.days
    team_rows["rest_days"] = team_rows["rest_days"].fillna(0.0)
    return team_rows


def _weighted_recent_points(prior_points: pd.Series) -> float:
    values = prior_points.tail(5).to_numpy(dtype=np.float32, copy=True)
    if len(values) == 0:
        return 0.0
    weights = np.arange(1, len(values) + 1, dtype=np.float32)
    return float(np.average(values, weights=weights))


def _prior_streaks(history: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
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
        if points > 0:
            unbeaten_streak += 1
        else:
            unbeaten_streak = 0
        if points == 3:
            win_streak += 1
        else:
            win_streak = 0
        if points == 0:
            losing_streak += 1
        else:
            losing_streak = 0

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
    sum_metrics = {
        "points",
        "wins",
        "draws",
        "losses",
        "clean_sheets",
        "failed_to_score",
    }

    for metric_name, source_col in rolling_sources.items():
        shifted = grouped[source_col].shift(1)
        for window in ROLLING_WINDOWS:
            rolling = shifted.groupby(df["team"]).rolling(window=window, min_periods=1)
            if metric_name in sum_metrics:
                output_col = f"{metric_name}_sum_last_{window}"
                df[output_col] = rolling.sum().reset_index(level=0, drop=True)
            if metric_name == "points" or metric_name not in sum_metrics:
                output_col = f"{metric_name}_avg_last_{window}"
                df[output_col] = rolling.mean().reset_index(level=0, drop=True)

    weighted_form = []
    streak_frames = []
    for _, history in df.groupby("team", sort=False):
        history = history.sort_values(["Date", "match_id"])
        weighted_form.extend(
            _weighted_recent_points(history["points"].iloc[:index])
            for index in range(len(history))
        )
        streak_frames.append(_prior_streaks(history))

    df["recent_form_points_weighted"] = weighted_form
    streaks = pd.concat(streak_frames, ignore_index=True)
    df = df.merge(streaks, on=["match_id", "team"], how="left")
    for feature in AUGMENTED_SEQUENCE_FEATURES:
        if feature not in df.columns:
            raise ValueError(f"Missing augmented sequence feature: {feature}")
    df[AUGMENTED_SEQUENCE_FEATURES] = df[AUGMENTED_SEQUENCE_FEATURES].fillna(0.0)
    return df


def sequence_features_for_mode(feature_mode: str) -> list[str]:
    if feature_mode == RAW_FEATURE_MODE:
        return SEQUENCE_FEATURES
    if feature_mode == RAW_PLUS_ROLLING_FEATURE_MODE:
        return AUGMENTED_SEQUENCE_FEATURES
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def prepare_team_rows_for_feature_mode(
    matches: pd.DataFrame,
    feature_mode: str,
) -> pd.DataFrame:
    team_rows = build_sequence_team_rows(matches)
    elo_features = team_elo_features(compute_elo_ratings(matches))
    team_rows = team_rows.merge(
        elo_features[["match_id", "team", "opponent", "team_elo"]],
        on=["match_id", "team", "opponent"],
        how="left",
    )
    if feature_mode == RAW_FEATURE_MODE:
        return team_rows
    if feature_mode == RAW_PLUS_ROLLING_FEATURE_MODE:
        return add_shifted_rolling_sequence_features(team_rows)
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def build_team_form_sequences(
    matches: pd.DataFrame,
    team_column: str,
    sort_column: str,
    feature_columns: list[str],
    sequence_length: int,
) -> dict[int, pd.DataFrame]:
    """Build strict-prior team-perspective windows keyed by target match row.

    This compatibility helper is used by the LSTM pipeline. It intentionally
    derives features from prior team-perspective match rows and never includes
    the target fixture row in the returned sequence.
    """

    if team_column not in {"HomeTeam", "AwayTeam"}:
        raise ValueError(f"Unsupported team column: {team_column}")
    if sort_column != "Date":
        raise ValueError("build_team_form_sequences expects Date ordering")

    working = matches.copy()
    if "match_id" not in working.columns:
        working["match_id"] = working.index
    if "season" not in working.columns:
        working["season"] = "unknown"
    working["Date"] = pd.to_datetime(working["Date"], errors="raise")

    team_rows = prepare_team_rows_for_feature_mode(working, RAW_FEATURE_MODE)
    histories = _build_team_history_index(team_rows)

    missing_features = [col for col in feature_columns if col not in team_rows.columns]
    if missing_features:
        raise ValueError(f"Missing sequence feature columns: {missing_features}")

    sequences: dict[int, pd.DataFrame] = {}
    for match in working.sort_values(["Date", "match_id"]).itertuples():
        team_name = getattr(match, team_column)
        history = histories.get(team_name)
        if history is None:
            continue
        window = _strict_prior_history(history, match.Date, sequence_length)
        if window is None:
            continue
        sequences[int(match.Index)] = window[feature_columns].copy()

    return sequences


def _split_bounds(raw_dir: Path = RAW_DIR) -> tuple[pd.Timestamp, pd.Timestamp]:
    raw_split_paths = require_raw_split_files(raw_dir)
    raw_train = pd.read_csv(raw_split_paths["train"])
    raw_val = pd.read_csv(raw_split_paths["val"])
    raw_train["Date"] = parse_match_dates(raw_train["Date"])
    raw_val["Date"] = parse_match_dates(raw_val["Date"])
    return raw_train["Date"].max(), raw_val["Date"].max()


def _split_name_for_date(
    target_date: pd.Timestamp,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
) -> str:
    if target_date <= train_end:
        return "train"
    if target_date <= val_end:
        return "val"
    return "test"


def _build_team_history_index(team_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    for team, history in team_rows.groupby("team", sort=False):
        histories[team] = history.sort_values(["Date", "match_id"]).reset_index(drop=True)
    return histories


def _strict_prior_history(
    history: pd.DataFrame,
    target_date: pd.Timestamp,
    sequence_length: int,
) -> pd.DataFrame | None:
    prior_matches = history[history["Date"] < target_date]
    if len(prior_matches) < sequence_length:
        return None
    return prior_matches.tail(sequence_length).copy()


def build_home_only_sequences(
    sequence_length: int = SEQUENCE_LENGTH,
    feature_mode: str = RAW_FEATURE_MODE,
) -> HomeOnlySequenceBuildResult:
    matches = load_raw_matches().sort_values(["Date", "season", "HomeTeam", "AwayTeam"])
    matches = matches.reset_index(drop=True)
    feature_names = sequence_features_for_mode(feature_mode)
    team_rows = prepare_team_rows_for_feature_mode(matches, feature_mode)
    histories = _build_team_history_index(team_rows)
    train_end, val_end = _split_bounds()

    split_arrays: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    split_labels: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    metadata_rows: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    leakage_check_passed = True

    for fixture in matches.itertuples(index=False):
        split_name = _split_name_for_date(fixture.Date, train_end, val_end)
        home_history = histories[fixture.HomeTeam]
        window = _strict_prior_history(home_history, fixture.Date, sequence_length)
        if window is None:
            continue

        history_end_date = window["Date"].max()
        if not bool(history_end_date < fixture.Date):
            leakage_check_passed = False
            raise ValueError(
                "Sequence leakage detected: target fixture includes a non-prior home-team match."
            )

        sequence = window[feature_names].to_numpy(dtype=np.float32, copy=True)
        split_arrays[split_name].append(sequence)
        split_labels[split_name].append(int(TARGET_MAP[fixture.FTR]))
        metadata_rows[split_name].append(
            {
                "match_id": int(fixture.match_id),
                "Date": fixture.Date,
                "HomeTeam": fixture.HomeTeam,
                "AwayTeam": fixture.AwayTeam,
                "FTR": fixture.FTR,
                "target": int(TARGET_MAP[fixture.FTR]),
                "target_label": TARGET_NAMES[TARGET_MAP[fixture.FTR]],
                "history_start_date": window["Date"].min(),
                "history_end_date": history_end_date,
                "split": split_name,
            }
        )

    num_features = len(feature_names)
    splits: dict[str, SequenceSplit] = {}
    for split_name in ("train", "val", "test"):
        metadata = pd.DataFrame(metadata_rows[split_name]).sort_values("Date").reset_index(
            drop=True
        )
        if split_arrays[split_name]:
            X = np.stack(split_arrays[split_name]).astype(np.float32, copy=False)
            y = np.asarray(split_labels[split_name], dtype=np.int64)
        else:
            X = np.empty((0, sequence_length, num_features), dtype=np.float32)
            y = np.empty((0,), dtype=np.int64)
        splits[split_name] = SequenceSplit(X=X, y=y, metadata=metadata)

    return HomeOnlySequenceBuildResult(
        feature_names=feature_names,
        sequence_length=sequence_length,
        leakage_check_passed=leakage_check_passed,
        splits=splits,
    )


def build_home_away_sequences(
    sequence_length: int = SEQUENCE_LENGTH,
    feature_mode: str = RAW_FEATURE_MODE,
) -> HomeOnlySequenceBuildResult:
    matches = load_raw_matches().sort_values(["Date", "season", "HomeTeam", "AwayTeam"])
    matches = matches.reset_index(drop=True)
    base_feature_names = sequence_features_for_mode(feature_mode)
    team_rows = prepare_team_rows_for_feature_mode(matches, feature_mode)
    histories = _build_team_history_index(team_rows)
    train_end, val_end = _split_bounds()

    home_feature_names = [f"home_{feature}" for feature in base_feature_names]
    away_feature_names = [f"away_{feature}" for feature in base_feature_names]
    feature_names = [*home_feature_names, *away_feature_names]

    split_arrays: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    split_labels: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    metadata_rows: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    leakage_check_passed = True

    for fixture in matches.itertuples(index=False):
        split_name = _split_name_for_date(fixture.Date, train_end, val_end)
        home_window = _strict_prior_history(histories[fixture.HomeTeam], fixture.Date, sequence_length)
        away_window = _strict_prior_history(histories[fixture.AwayTeam], fixture.Date, sequence_length)
        if home_window is None or away_window is None:
            continue

        home_history_end_date = home_window["Date"].max()
        away_history_end_date = away_window["Date"].max()
        if not bool(home_history_end_date < fixture.Date and away_history_end_date < fixture.Date):
            leakage_check_passed = False
            raise ValueError(
                "Sequence leakage detected: target fixture includes a non-prior team history row."
            )

        home_sequence = home_window[base_feature_names].to_numpy(dtype=np.float32, copy=True)
        away_sequence = away_window[base_feature_names].to_numpy(dtype=np.float32, copy=True)
        combined_sequence = np.concatenate([home_sequence, away_sequence], axis=1)

        split_arrays[split_name].append(combined_sequence)
        split_labels[split_name].append(int(TARGET_MAP[fixture.FTR]))
        metadata_rows[split_name].append(
            {
                "match_id": int(fixture.match_id),
                "Date": fixture.Date,
                "HomeTeam": fixture.HomeTeam,
                "AwayTeam": fixture.AwayTeam,
                "FTR": fixture.FTR,
                "target": int(TARGET_MAP[fixture.FTR]),
                "target_label": TARGET_NAMES[TARGET_MAP[fixture.FTR]],
                "home_history_start_date": home_window["Date"].min(),
                "home_history_end_date": home_history_end_date,
                "away_history_start_date": away_window["Date"].min(),
                "away_history_end_date": away_history_end_date,
                "split": split_name,
            }
        )

    num_features = len(feature_names)
    splits: dict[str, SequenceSplit] = {}
    for split_name in ("train", "val", "test"):
        metadata = pd.DataFrame(metadata_rows[split_name]).sort_values("Date").reset_index(
            drop=True
        )
        if split_arrays[split_name]:
            X = np.stack(split_arrays[split_name]).astype(np.float32, copy=False)
            y = np.asarray(split_labels[split_name], dtype=np.int64)
        else:
            X = np.empty((0, sequence_length, num_features), dtype=np.float32)
            y = np.empty((0,), dtype=np.int64)
        splits[split_name] = SequenceSplit(X=X, y=y, metadata=metadata)

    return HomeOnlySequenceBuildResult(
        feature_names=feature_names,
        sequence_length=sequence_length,
        leakage_check_passed=leakage_check_passed,
        splits=splits,
    )


def target_distribution(y: np.ndarray) -> pd.Series:
    return (
        pd.Series(y, dtype="int64")
        .value_counts()
        .sort_index()
        .reindex([0, 1, 2], fill_value=0)
        .rename(index={0: "0 HomeWin", 1: "1 Draw", 2: "2 AwayWin"})
    )


def save_home_only_sequences(
    result: HomeOnlySequenceBuildResult,
    output_dir: Path = SEQUENCE_OUTPUT_DIR,
    prefix: str = HOME_ONLY_PREFIX,
) -> dict[str, tuple[Path, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, tuple[Path, Path]] = {}
    feature_names = np.asarray(result.feature_names)
    (output_dir / "feature_names.json").write_text(
        json.dumps(result.feature_names, indent=2),
        encoding="utf-8",
    )

    for split_name, split in result.splits.items():
        npz_path = output_dir / f"{prefix}_{split_name}.npz"
        metadata_path = output_dir / f"{prefix}_{split_name}_metadata.csv"
        np.savez(
            npz_path,
            X=split.X,
            y=split.y,
            feature_names=feature_names,
        )
        split.metadata.to_csv(metadata_path, index=False)
        saved_paths[split_name] = (npz_path, metadata_path)

    return saved_paths


def build_summary(
    result: HomeOnlySequenceBuildResult,
    saved_paths: dict[str, tuple[Path, Path]] | None = None,
    title: str = "Home-Only Sequence Build Summary",
) -> str:
    lines = [
        title,
        "=" * 32,
        f"Sequence length: {result.sequence_length}",
        f"Features per timestep: {len(result.feature_names)}",
        f"Feature names: {', '.join(result.feature_names)}",
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
                "Date range: "
                f"{split.metadata['Date'].min().date()} to "
                f"{split.metadata['Date'].max().date()}"
            )
        else:
            lines.append("Date range: n/a")
        lines.append("Target distribution:")
        lines.append(target_distribution(split.y).to_string())
        if saved_paths is not None:
            npz_path, metadata_path = saved_paths[split_name]
            lines.append(f"Saved npz: {npz_path}")
            lines.append(f"Saved metadata: {metadata_path}")
        lines.append("")

    return "\n".join(lines).rstrip()


def main() -> None:
    home_only_result = build_home_only_sequences()
    home_only_paths = save_home_only_sequences(home_only_result, prefix=HOME_ONLY_PREFIX)
    print(build_summary(home_only_result, home_only_paths, title="Home-Only Sequence Build Summary"))
    print()

    home_away_result = build_home_away_sequences()
    home_away_paths = save_home_only_sequences(home_away_result, prefix=HOME_AWAY_PREFIX)
    print(build_summary(home_away_result, home_away_paths, title="Home-Away Sequence Build Summary"))


if __name__ == "__main__":
    main()
