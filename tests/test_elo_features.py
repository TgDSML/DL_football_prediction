from __future__ import annotations

import pandas as pd

from src.features.elo import compute_elo_ratings


def test_elo_uses_pre_match_rating_before_update() -> None:
    matches = pd.DataFrame(
        {
            "match_id": [0, 1],
            "Date": pd.to_datetime(["2020-01-01", "2020-01-08"]),
            "season": ["2020", "2020"],
            "HomeTeam": ["A", "A"],
            "AwayTeam": ["B", "B"],
            "FTR": ["H", "D"],
        }
    )

    elo = compute_elo_ratings(matches, k_factor=20)

    first = elo.iloc[0]
    second = elo.iloc[1]
    assert first["home_elo"] == 1500.0
    assert first["away_elo"] == 1500.0
    assert first["home_post_elo"] == 1510.0
    assert first["away_post_elo"] == 1490.0
    assert second["home_elo"] == first["home_post_elo"]
    assert second["away_elo"] == first["away_post_elo"]


def test_draw_updates_both_teams_against_expected_score() -> None:
    matches = pd.DataFrame(
        {
            "match_id": [0],
            "Date": pd.to_datetime(["2020-01-01"]),
            "season": ["2020"],
            "HomeTeam": ["A"],
            "AwayTeam": ["B"],
            "FTR": ["D"],
        }
    )

    elo = compute_elo_ratings(matches, k_factor=20).iloc[0]

    assert elo["home_elo"] == 1500.0
    assert elo["away_elo"] == 1500.0
    assert elo["home_post_elo"] == 1500.0
    assert elo["away_post_elo"] == 1500.0
