from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


INITIAL_ELO = 1500.0
DEFAULT_K_FACTOR = 20.0


@dataclass(frozen=True)
class EloConfig:
    initial_rating: float = INITIAL_ELO
    k_factor: float = DEFAULT_K_FACTOR


def expected_score(rating: float, opponent_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))


def actual_home_score(result: str) -> float:
    if result == "H":
        return 1.0
    if result == "D":
        return 0.5
    if result == "A":
        return 0.0
    raise ValueError(f"Unsupported full-time result for Elo: {result!r}")


def compute_elo_ratings(
    matches: pd.DataFrame,
    *,
    initial_rating: float = INITIAL_ELO,
    k_factor: float = DEFAULT_K_FACTOR,
) -> pd.DataFrame:
    required_columns = {"match_id", "Date", "HomeTeam", "AwayTeam", "FTR"}
    missing = sorted(required_columns.difference(matches.columns))
    if missing:
        raise ValueError(f"Cannot compute Elo ratings; missing columns: {missing}")

    ratings: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    ordered = matches.sort_values(["Date", "season", "HomeTeam", "AwayTeam"])

    for match in ordered.itertuples(index=False):
        home_team = str(match.HomeTeam)
        away_team = str(match.AwayTeam)
        home_elo = ratings.get(home_team, initial_rating)
        away_elo = ratings.get(away_team, initial_rating)

        expected_home = expected_score(home_elo, away_elo)
        expected_away = 1.0 - expected_home
        actual_home = actual_home_score(str(match.FTR))
        actual_away = 1.0 - actual_home

        home_post_elo = home_elo + k_factor * (actual_home - expected_home)
        away_post_elo = away_elo + k_factor * (actual_away - expected_away)

        rows.append(
            {
                "match_id": int(match.match_id),
                "Date": match.Date,
                "HomeTeam": home_team,
                "AwayTeam": away_team,
                "FTR": match.FTR,
                "home_elo": home_elo,
                "away_elo": away_elo,
                "elo_diff": home_elo - away_elo,
                "home_post_elo": home_post_elo,
                "away_post_elo": away_post_elo,
                "expected_home": expected_home,
                "expected_away": expected_away,
            }
        )

        ratings[home_team] = home_post_elo
        ratings[away_team] = away_post_elo

    return pd.DataFrame(rows).sort_values("match_id").reset_index(drop=True)


def fixture_elo_features(elo_ratings: pd.DataFrame) -> pd.DataFrame:
    return elo_ratings[["match_id", "home_elo", "away_elo", "elo_diff"]].copy()


def team_elo_features(elo_ratings: pd.DataFrame) -> pd.DataFrame:
    home = pd.DataFrame(
        {
            "match_id": elo_ratings["match_id"],
            "team": elo_ratings["HomeTeam"],
            "opponent": elo_ratings["AwayTeam"],
            "team_elo": elo_ratings["home_elo"],
            "opponent_elo": elo_ratings["away_elo"],
            "elo_diff": elo_ratings["elo_diff"],
        }
    )
    away = pd.DataFrame(
        {
            "match_id": elo_ratings["match_id"],
            "team": elo_ratings["AwayTeam"],
            "opponent": elo_ratings["HomeTeam"],
            "team_elo": elo_ratings["away_elo"],
            "opponent_elo": elo_ratings["home_elo"],
            "elo_diff": -elo_ratings["elo_diff"],
        }
    )
    return pd.concat([home, away], ignore_index=True)
