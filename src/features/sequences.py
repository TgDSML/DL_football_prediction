from __future__ import annotations

import pandas as pd


def build_team_form_sequences(
    matches: pd.DataFrame,
    team_column: str,
    sort_column: str,
    feature_columns: list[str],
    sequence_length: int,
) -> dict[int, pd.DataFrame]:
    """Build rolling team-form windows aligned to actual match rows.

    For each match row, return the prior sequence of feature values for the team.
    This keeps home and away team history aligned to the same match.
    """

    sequences: dict[int, pd.DataFrame] = {}
    for team, team_matches in matches.groupby(team_column):
        team_matches_sorted = team_matches.sort_values(sort_column)
        features = team_matches_sorted[feature_columns]
        for end_index in range(sequence_length, len(features)):
            match_idx = team_matches_sorted.index[end_index]
            sequences[int(match_idx)] = features.iloc[end_index - sequence_length : end_index]
    return sequences
