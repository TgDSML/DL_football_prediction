import os
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

INPUT_FILES = {
    "train": os.path.join(INTERIM_DIR, "train_processed.csv"),
    "val": os.path.join(INTERIM_DIR, "val_processed.csv"),
    "test": os.path.join(INTERIM_DIR, "test_processed.csv"),
}

OUTPUT_FILES = {
    "train": os.path.join(PROCESSED_DIR, "train_team_centric.csv"),
    "val": os.path.join(PROCESSED_DIR, "val_team_centric.csv"),
    "test": os.path.join(PROCESSED_DIR, "test_team_centric.csv"),
}

DATE_COL = "Date"
SEASON_COL = "season"
HOME_TEAM_COL = "HomeTeam"
AWAY_TEAM_COL = "AwayTeam"
REFEREE_COL = "Referee"
RESULT_COL = "FTR"
HOME_GOALS_COL = "FTHG"
AWAY_GOALS_COL = "FTAG"


def build_team_side(df, side, context_cols):
    if side == "home":
        team_col = HOME_TEAM_COL
        opp_col = AWAY_TEAM_COL
        gf_col = HOME_GOALS_COL
        ga_col = AWAY_GOALS_COL
        role = 1
        target_map = {2: 2, 1: 1, 0: 0}
    else:
        team_col = AWAY_TEAM_COL
        opp_col = HOME_TEAM_COL
        gf_col = AWAY_GOALS_COL
        ga_col = HOME_GOALS_COL
        role = 0
        target_map = {0: 2, 1: 1, 2: 0}

    side_df = df[
        context_cols + ["match_id", team_col, opp_col, gf_col, ga_col, RESULT_COL]
    ].copy()

    side_df = side_df.rename(columns={
        team_col: "team",
        opp_col: "opponent",
        gf_col: "gf",
        ga_col: "ga"
    })

    side_df["role"] = role
    side_df["target"] = side_df[RESULT_COL].map(target_map)
    side_df["points"] = side_df["target"].map({2: 3, 1: 1, 0: 0})
    side_df["win"] = (side_df["target"] == 2).astype(int)
    side_df["draw"] = (side_df["target"] == 1).astype(int)
    side_df["loss"] = (side_df["target"] == 0).astype(int)

    return side_df


def build_team_centric(df):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([SEASON_COL, DATE_COL]).reset_index(drop=True)
    df["match_id"] = range(len(df))

    context_cols = [
        col for col in [
            DATE_COL, SEASON_COL, "year", "month", "day_of_week", "is_weekend", REFEREE_COL
        ]
        if col in df.columns
    ]

    home_df = build_team_side(df, side="home", context_cols=context_cols)
    away_df = build_team_side(df, side="away", context_cols=context_cols)

    team_df = pd.concat([home_df, away_df], ignore_index=True)
    team_df = team_df.sort_values(
        [SEASON_COL, DATE_COL, "match_id", "role"],
        ascending=[True, True, True, False]
    ).reset_index(drop=True)

    return team_df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    for split_name, input_path in INPUT_FILES.items():
        if not os.path.exists(input_path):
            print(f"Skipping missing file: {input_path}")
            continue

        df = pd.read_csv(input_path)
        team_df = build_team_centric(df)

        out_path = OUTPUT_FILES[split_name]
        team_df.to_csv(out_path, index=False)

        print(f"Saved {split_name}: {len(team_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()