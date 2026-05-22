from __future__ import annotations

import pandas as pd


def build_team_form_sequences(
    matches: pd.DataFrame,
    team_column: str,
    sort_column: str,
    feature_columns: list[str],
    sequence_length: int,
) -> list[pd.DataFrame]:
    """Build rolling team-form windows.

    This placeholder returns one DataFrame per complete historical window.
    Later, pair home and away windows into model-ready tensors.
    """

    sequences: list[pd.DataFrame] = []
    ordered = matches.sort_values([team_column, sort_column])
    for _, team_matches in ordered.groupby(team_column):
        features = team_matches[feature_columns]
        for end_index in range(sequence_length, len(features) + 1):
            sequences.append(features.iloc[end_index - sequence_length : end_index])
    return sequences
