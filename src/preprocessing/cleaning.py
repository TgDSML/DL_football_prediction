from __future__ import annotations

import pandas as pd


def clean_matches(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of raw match data."""

    cleaned = frame.copy()
    cleaned.columns = [column.strip().lower().replace(" ", "_") for column in cleaned.columns]
    return cleaned
