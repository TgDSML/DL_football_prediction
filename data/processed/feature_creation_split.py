import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

INPUT_FILES = {
    "train": os.path.join(PROCESSED_DIR, "train_team_centric.csv"),
    "val": os.path.join(PROCESSED_DIR, "val_team_centric.csv"),
    "test": os.path.join(PROCESSED_DIR, "test_team_centric.csv"),
}

OUTPUT_FILES = {
    "train": os.path.join(PROCESSED_DIR, "train_features.csv"),
    "val": os.path.join(PROCESSED_DIR, "val_features.csv"),
    "test": os.path.join(PROCESSED_DIR, "test_features.csv"),
}

OUTPUT_COMBINED_PATH = os.path.join(PROCESSED_DIR, "02_features.csv")

DATE_COL = "Date"
SEASON_COL = "season"
WINDOWS = [3, 5, 10]

ELO_BASE = 1500.0
ELO_K = 20.0
ELO_CARRYOVER = 0.75
EARLY_SEASON_MATCHES = 5


def mean_or_nan(values):
    return float(np.mean(values)) if len(values) > 0 else np.nan


def var_or_nan(values):
    return float(np.var(values, ddof=1)) if len(values) > 1 else np.nan


class TeamHistoryState:
    def __init__(self, windows):
        self.windows = windows
        self.points = {w: defaultdict(lambda: deque(maxlen=w)) for w in windows}
        self.wins = {w: defaultdict(lambda: deque(maxlen=w)) for w in windows}
        self.gf = {w: defaultdict(lambda: deque(maxlen=w)) for w in windows}
        self.ga = {w: defaultdict(lambda: deque(maxlen=w)) for w in windows}

        self.last_date = {}
        self.season_team_match_count = defaultdict(int)

        self.prev_season_summary = {}
        self.last_seen_season = {}

        self.team_elo = {}
        self.current_season = None

    def transition_season_if_needed(self, season):
        if self.current_season is None:
            self.current_season = season
            return

        if season == self.current_season:
            return

        completed_season = self.current_season

        season_totals = {}
        teams_in_season = set()
        for w in self.windows:
            teams_in_season.update(self.points[w].keys())

        for team in teams_in_season:
            pts_hist = list(self.points[max(self.windows)].get(team, []))
            gf_hist = list(self.gf[max(self.windows)].get(team, []))
            ga_hist = list(self.ga[max(self.windows)].get(team, []))
            played = self.season_team_match_count.get((completed_season, team), 0)

            if played > 0:
                total_points = sum(pts_hist) if len(pts_hist) == played else None
                total_gf = sum(gf_hist) if len(gf_hist) == played else None
                total_ga = sum(ga_hist) if len(ga_hist) == played else None

        exact_points = defaultdict(int)
        exact_gf = defaultdict(int)
        exact_ga = defaultdict(int)
        exact_games = defaultdict(int)

        for (season_key, team), games in self.season_team_match_count.items():
            if season_key != completed_season:
                continue
            exact_games[team] = games

        # These totals are filled incrementally in update_after_match via dedicated dicts
        for team, games in exact_games.items():
            season_totals[team] = {
                "points_per_game": self.prev_season_running_points[(completed_season, team)] / games,
                "goals_for_per_game": self.prev_season_running_gf[(completed_season, team)] / games,
                "goals_against_per_game": self.prev_season_running_ga[(completed_season, team)] / games,
            }

        self.prev_season_summary = season_totals

        new_elo = {}
        for team, rating in self.team_elo.items():
            new_elo[team] = ELO_BASE + ELO_CARRYOVER * (rating - ELO_BASE)
        self.team_elo = new_elo

        self.current_season = season

    def initialize_running_totals(self):
        self.prev_season_running_points = defaultdict(float)
        self.prev_season_running_gf = defaultdict(float)
        self.prev_season_running_ga = defaultdict(float)

    def get_team_features(self, team, date, season):
        feats = {}

        last_date = self.last_date.get(team)
        days_rest = (date - last_date).days if last_date is not None else np.nan
        feats["days_rest"] = days_rest
        feats["match_density"] = 1.0 / days_rest if pd.notna(days_rest) and days_rest > 0 else np.nan

        team_match_number = self.season_team_match_count[(season, team)] + 1
        feats["team_match_number"] = team_match_number
        feats["is_early_season"] = int(team_match_number <= EARLY_SEASON_MATCHES)

        prev_summary = self.prev_season_summary.get(team, {})
        feats["prev_season_points_per_game"] = prev_summary.get("points_per_game", np.nan)
        feats["prev_season_goals_for_per_game"] = prev_summary.get("goals_for_per_game", np.nan)
        feats["prev_season_goals_against_per_game"] = prev_summary.get("goals_against_per_game", np.nan)

        last_season_seen = self.last_seen_season.get(team)
        feats["is_promoted_cold_start"] = int(
            team_match_number == 1 and (last_season_seen is None or last_season_seen != season - 1)
        )

        feats["team_elo"] = self.team_elo.get(team, ELO_BASE)

        for w in self.windows:
            pts = list(self.points[w][team])
            wins = list(self.wins[w][team])
            gf = list(self.gf[w][team])
            ga = list(self.ga[w][team])

            feats[f"avg_form_{w}"] = mean_or_nan(pts)
            feats[f"win_rate_{w}"] = mean_or_nan(wins)
            feats[f"gf_avg_{w}"] = mean_or_nan(gf)
            feats[f"ga_avg_{w}"] = mean_or_nan(ga)
            feats[f"goal_var_{w}"] = var_or_nan(gf)
            feats[f"goal_balance_{w}"] = feats[f"gf_avg_{w}"] - feats[f"ga_avg_{w}"] if pd.notna(feats[f"gf_avg_{w}"]) and pd.notna(feats[f"ga_avg_{w}"]) else np.nan

            short_w = max(2, w // 2)
            short_pts = pts[-short_w:] if len(pts) > 0 else []
            short_form = mean_or_nan(short_pts)
            long_form = feats[f"avg_form_{w}"]
            feats[f"form_momentum_{w}"] = short_form - long_form if pd.notna(short_form) and pd.notna(long_form) else np.nan

        return feats

    def update_after_match(self, season, date, home_team, away_team, home_target, home_gf, home_ga):
        if home_target == 2:
            home_points, away_points = 3, 0
            home_win, away_win = 1, 0
            score_home = 1.0
        elif home_target == 1:
            home_points, away_points = 1, 1
            home_win, away_win = 0, 0
            score_home = 0.5
        else:
            home_points, away_points = 0, 3
            home_win, away_win = 0, 1
            score_home = 0.0

        score_away = 1.0 - score_home

        home_elo = self.team_elo.get(home_team, ELO_BASE)
        away_elo = self.team_elo.get(away_team, ELO_BASE)
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        expected_away = 1.0 - expected_home

        self.team_elo[home_team] = home_elo + ELO_K * (score_home - expected_home)
        self.team_elo[away_team] = away_elo + ELO_K * (score_away - expected_away)

        away_gf, away_ga = home_ga, home_gf

        for w in self.windows:
            self.points[w][home_team].append(home_points)
            self.points[w][away_team].append(away_points)

            self.wins[w][home_team].append(home_win)
            self.wins[w][away_team].append(away_win)

            self.gf[w][home_team].append(home_gf)
            self.gf[w][away_team].append(away_gf)

            self.ga[w][home_team].append(home_ga)
            self.ga[w][away_team].append(away_ga)

        self.last_date[home_team] = date
        self.last_date[away_team] = date

        self.season_team_match_count[(season, home_team)] += 1
        self.season_team_match_count[(season, away_team)] += 1

        self.prev_season_running_points[(season, home_team)] += home_points
        self.prev_season_running_points[(season, away_team)] += away_points
        self.prev_season_running_gf[(season, home_team)] += home_gf
        self.prev_season_running_gf[(season, away_team)] += away_gf
        self.prev_season_running_ga[(season, home_team)] += home_ga
        self.prev_season_running_ga[(season, away_team)] += away_ga

        self.last_seen_season[home_team] = season
        self.last_seen_season[away_team] = season


def add_match_level_progress(df):
    season_order = (
        df[[SEASON_COL, DATE_COL, "match_id"]]
        .drop_duplicates()
        .sort_values([SEASON_COL, DATE_COL, "match_id"])
        .copy()
    )
    season_order["match_number_global"] = season_order.groupby(SEASON_COL).cumcount() + 1
    season_total = season_order.groupby(SEASON_COL)["match_id"].count().rename("season_total_matches").reset_index()
    season_order = season_order.merge(season_total, on=SEASON_COL, how="left")
    season_order["season_progress"] = season_order["match_number_global"] / season_order["season_total_matches"]

    df = df.merge(
        season_order[[SEASON_COL, "match_id", "season_progress"]],
        on=[SEASON_COL, "match_id"],
        how="left"
    )
    return df


def merge_opponent_features(df):
    base_cols = [
        "days_rest", "match_density",
        "team_elo",
        "prev_season_points_per_game",
        "prev_season_goals_for_per_game",
        "prev_season_goals_against_per_game",
        "is_promoted_cold_start",
    ]

    rolling_cols = []
    for w in WINDOWS:
        rolling_cols.extend([
            f"avg_form_{w}", f"win_rate_{w}", f"gf_avg_{w}", f"ga_avg_{w}",
            f"goal_var_{w}", f"goal_balance_{w}", f"form_momentum_{w}"
        ])

    feature_cols = base_cols + rolling_cols

    opp_df = df[["match_id", "team"] + feature_cols].copy()
    opp_df = opp_df.rename(columns={"team": "opponent", **{c: f"opp_{c}" for c in feature_cols}})

    return df.merge(opp_df, on=["match_id", "opponent"], how="left")


def process_split(df, split_name, state):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values([SEASON_COL, DATE_COL, "match_id", "role"], ascending=[True, True, True, False]).reset_index(drop=True)

    rows = []

    match_level = (
        df.groupby("match_id", sort=False)
        .apply(lambda g: g.sort_values("role", ascending=False))
        .reset_index(drop=True)
    )

    for match_id, grp in match_level.groupby("match_id", sort=False):
        if len(grp) != 2:
            continue

        grp = grp.sort_values("role", ascending=False)
        home_row = grp.iloc[0]
        away_row = grp.iloc[1]

        season = home_row[SEASON_COL]
        date = home_row[DATE_COL]

        state.transition_season_if_needed(season)

        home_team = home_row["team"]
        away_team = away_row["team"]

        home_feats = state.get_team_features(home_team, date, season)
        away_feats = state.get_team_features(away_team, date, season)

        home_out = home_row.to_dict()
        away_out = away_row.to_dict()

        home_out.update(home_feats)
        away_out.update(away_feats)

        rows.append(home_out)
        rows.append(away_out)

        state.update_after_match(
            season=season,
            date=date,
            home_team=home_team,
            away_team=away_team,
            home_target=home_row["target"],
            home_gf=home_row["gf"],
            home_ga=home_row["ga"]
        )

    out_df = pd.DataFrame(rows)
    out_df["split"] = split_name
    out_df = add_match_level_progress(out_df)
    out_df = merge_opponent_features(out_df)

    return out_df, state


def select_final_columns(df):
    final_cols = [
        "match_id", DATE_COL, SEASON_COL, "split",
        "team", "opponent", "target", "role",
        "month", "day_of_week", "is_weekend",
        "season_progress", "is_early_season",
        "days_rest", "match_density",
        "team_elo", "opp_team_elo",
        "prev_season_points_per_game",
        "prev_season_goals_for_per_game",
        "prev_season_goals_against_per_game",
        "is_promoted_cold_start",
        "opp_days_rest", "opp_match_density",
        "opp_prev_season_points_per_game",
        "opp_prev_season_goals_for_per_game",
        "opp_prev_season_goals_against_per_game",
        "opp_is_promoted_cold_start",
    ]

    for w in WINDOWS:
        final_cols.extend([
            f"avg_form_{w}", f"win_rate_{w}", f"gf_avg_{w}", f"ga_avg_{w}",
            f"goal_var_{w}", f"goal_balance_{w}", f"form_momentum_{w}",
            f"opp_avg_form_{w}", f"opp_win_rate_{w}", f"opp_gf_avg_{w}",
            f"opp_ga_avg_{w}", f"opp_goal_var_{w}",
            f"opp_goal_balance_{w}", f"opp_form_momentum_{w}",
        ])

    df["elo_advantage"] = df["team_elo"] - df["opp_team_elo"]
    final_cols.insert(final_cols.index("prev_season_points_per_game"), "elo_advantage")

    final_cols = [c for c in final_cols if c in df.columns]
    return df[final_cols].copy()


def main():
    state = TeamHistoryState(WINDOWS)
    state.initialize_running_totals()

    all_frames = []

    for split_name in ["train", "val", "test"]:
        path = INPUT_FILES[split_name]
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue

        split_df = pd.read_csv(path)
        feat_df, state = process_split(split_df, split_name, state)
        feat_df = select_final_columns(feat_df)

        feat_df.to_csv(OUTPUT_FILES[split_name], index=False)
        all_frames.append(feat_df)

        print(f"Saved {split_name}: {len(feat_df)} rows -> {OUTPUT_FILES[split_name]}")

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_csv(OUTPUT_COMBINED_PATH, index=False)
        print(f"Saved combined features: {OUTPUT_COMBINED_PATH}")
        print(f"Rows: {len(combined)}")
        print(f"Columns: {combined.columns.tolist()}")


if __name__ == "__main__":
    main()