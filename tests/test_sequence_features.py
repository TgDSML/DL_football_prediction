from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.sequences import (
    RAW_FEATURE_MODE,
    RAW_PLUS_ROLLING_FEATURE_MODE,
    SEQUENCE_FEATURES,
    HomeOnlySequenceBuildResult,
    SequenceSplit,
    _strict_prior_history,
    add_shifted_rolling_sequence_features,
    save_home_only_sequences,
    sequence_features_for_mode,
)


def _minimal_team_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_id": [1, 2, 3],
            "Date": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"]),
            "season": ["2020", "2020", "2020"],
            "team": ["A", "A", "A"],
            "opponent": ["B", "C", "D"],
            "was_home": [1.0, 0.0, 1.0],
            "points": [3.0, 1.0, 0.0],
            "goals_for": [2.0, 1.0, 0.0],
            "goals_against": [0.0, 1.0, 1.0],
            "goal_diff": [2.0, 0.0, -1.0],
            "shots_for": [10.0, 8.0, 5.0],
            "shots_against": [4.0, 7.0, 9.0],
            "shots_on_target_for": [5.0, 3.0, 1.0],
            "shots_on_target_against": [1.0, 2.0, 4.0],
            "corners_for": [6.0, 4.0, 2.0],
            "corners_against": [2.0, 5.0, 7.0],
            "yellow_cards_for": [1.0, 2.0, 3.0],
            "yellow_cards_against": [2.0, 1.0, 1.0],
            "red_cards_for": [0.0, 0.0, 1.0],
            "red_cards_against": [0.0, 1.0, 0.0],
            "wins": [1.0, 0.0, 0.0],
            "draws": [0.0, 1.0, 0.0],
            "losses": [0.0, 0.0, 1.0],
            "clean_sheets": [1.0, 0.0, 0.0],
            "failed_to_score": [0.0, 0.0, 1.0],
            "rest_days": [0.0, 7.0, 7.0],
            "team_elo": [1500.0, 1510.0, 1510.0],
        }
    )


def test_raw_feature_mode_output_is_unchanged() -> None:
    assert sequence_features_for_mode(RAW_FEATURE_MODE) == SEQUENCE_FEATURES
    assert len(sequence_features_for_mode(RAW_FEATURE_MODE)) == 17


def test_raw_plus_rolling_feature_count_greater_than_raw() -> None:
    assert len(sequence_features_for_mode(RAW_PLUS_ROLLING_FEATURE_MODE)) > len(
        sequence_features_for_mode(RAW_FEATURE_MODE)
    )


def test_strict_prior_history_excludes_target_fixture_date() -> None:
    history = _minimal_team_rows()
    target_date = pd.Timestamp("2020-01-15")

    prior = _strict_prior_history(history, target_date, sequence_length=2)

    assert prior is not None
    assert prior["Date"].max() < target_date
    assert 3 not in prior["match_id"].tolist()


def test_rolling_features_for_timestep_do_not_include_that_match() -> None:
    rolled = add_shifted_rolling_sequence_features(_minimal_team_rows())

    first = rolled.iloc[0]
    second = rolled.iloc[1]
    third = rolled.iloc[2]

    assert first["points_sum_last_3"] == 0.0
    assert second["points_sum_last_3"] == 3.0
    assert second["points_sum_last_3"] != 3.0 + second["points"]
    assert third["points_sum_last_3"] == 4.0


def test_saved_feature_names_length_matches_x_dimension() -> None:
    result = HomeOnlySequenceBuildResult(
        feature_names=["a", "b", "c"],
        sequence_length=2,
        leakage_check_passed=True,
        splits={
            split_name: SequenceSplit(
                X=np.zeros((1, 2, 3), dtype=np.float32),
                y=np.array([0], dtype=np.int64),
                metadata=pd.DataFrame({"Date": [pd.Timestamp("2020-01-01")]}),
            )
            for split_name in ("train", "val", "test")
        },
    )

    output_dir = Path(".tmp/sequence_feature_test")
    save_home_only_sequences(result, output_dir=output_dir, prefix="toy")
    arrays = np.load(output_dir / "toy_train.npz", allow_pickle=False)

    assert len(arrays["feature_names"]) == arrays["X"].shape[-1]
