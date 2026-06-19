import os
import json
from collections import defaultdict

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
OUTPUT_BASE_DIR = os.path.join(ARTIFACTS_DIR, "cnn")

INPUT_FILES = {
    "train": os.path.join(PROCESSED_DIR, "train_features.csv"),
    "val": os.path.join(PROCESSED_DIR, "val_features.csv"),
    "test": os.path.join(PROCESSED_DIR, "test_features.csv"),
}

SEQUENCE_LENGTHS = [3, 5, 10]
WINDOW_SIZES = [3, 5, 10]

CONTEXT_FEATURES = [
    "month",
    "day_of_week",
    "is_weekend",
    "season_progress",
    "is_early_season",
]

ROLLING_WINDOW_FEATURES = [
    "avg_form",
    "win_rate",
    "gf_avg",
    "ga_avg",
    "goal_var",
    "goal_balance",
    "form_momentum",
]

OPP_ROLLING_WINDOW_FEATURES = [
    "opp_avg_form",
    "opp_win_rate",
    "opp_gf_avg",
    "opp_ga_avg",
    "opp_goal_var",
    "opp_goal_balance",
    "opp_form_momentum",
]

TEAM_METRICS = [
    "days_rest",
    "match_density",
    "team_elo",
]

OPP_TEAM_METRICS = [
    "opp_days_rest",
    "opp_match_density",
    "opp_team_elo",
]

PRIOR_SEASON_FEATURES = [
    "prev_season_points_per_game",
    "prev_season_goals_for_per_game",
    "prev_season_goals_against_per_game",
    "is_promoted_cold_start",
]

OPP_PRIOR_SEASON_FEATURES = [
    "opp_prev_season_points_per_game",
    "opp_prev_season_goals_for_per_game",
    "opp_prev_season_goals_against_per_game",
    "opp_is_promoted_cold_start",
]

LABEL_MAP = {
    "A": 0,
    "D": 1,
    "H": 2,
    "0": 0,
    "1": 1,
    "2": 2,
    0: 0,
    1: 1,
    2: 2,
}


class CNNSequenceBuilder:
    """
    Build true historical sequences for 1D CNN models.

    Output per sample:
        sequences:     (num_samples, sequence_length, num_features)
        sequences_cnn: (num_samples, num_features, sequence_length)

    Each timestep is one historical match before the target match.
    Each timestep vector is:
        [home_team_history_features | away_team_history_features]
    """

    def __init__(self, sequence_length=5):
        self.sequence_length = sequence_length
        self.match_history = defaultdict(list)
        self.match_history_dates = defaultdict(list)

        self.single_team_feature_names = self._build_single_team_feature_names()
        self.combined_feature_names = (
            [f"home_hist__{f}" for f in self.single_team_feature_names] +
            [f"away_hist__{f}" for f in self.single_team_feature_names]
        )

    @staticmethod
    def _safe_float(value):
        if pd.isna(value):
            return np.nan
        try:
            return float(value)
        except Exception:
            return np.nan

    @staticmethod
    def _safe_date(value):
        if pd.isna(value):
            return None
        try:
            return pd.to_datetime(value).date()
        except Exception:
            return None

    def _build_single_team_feature_names(self):
        names = []

        for window in WINDOW_SIZES:
            for feat in ROLLING_WINDOW_FEATURES:
                names.append(f"{feat}_{window}")

        for window in WINDOW_SIZES:
            for feat in OPP_ROLLING_WINDOW_FEATURES:
                names.append(f"{feat}_{window}")

        names.extend(TEAM_METRICS)
        names.extend(OPP_TEAM_METRICS)
        names.extend(PRIOR_SEASON_FEATURES)
        names.extend(OPP_PRIOR_SEASON_FEATURES)
        names.extend(CONTEXT_FEATURES)
        names.append("elo_advantage")

        return names

    def _build_feature_vector(self, row):
        channels = []

        for window in WINDOW_SIZES:
            for feat in ROLLING_WINDOW_FEATURES:
                channels.append(self._safe_float(row.get(f"{feat}_{window}", np.nan)))

        for window in WINDOW_SIZES:
            for feat in OPP_ROLLING_WINDOW_FEATURES:
                channels.append(self._safe_float(row.get(f"{feat}_{window}", np.nan)))

        for feat in TEAM_METRICS:
            channels.append(self._safe_float(row.get(feat, np.nan)))

        for feat in OPP_TEAM_METRICS:
            channels.append(self._safe_float(row.get(feat, np.nan)))

        for feat in PRIOR_SEASON_FEATURES:
            channels.append(self._safe_float(row.get(feat, np.nan)))

        for feat in OPP_PRIOR_SEASON_FEATURES:
            channels.append(self._safe_float(row.get(feat, np.nan)))

        for feat in CONTEXT_FEATURES:
            channels.append(self._safe_float(row.get(feat, np.nan)))

        team_elo = self._safe_float(row.get("team_elo", np.nan))
        opp_team_elo = self._safe_float(row.get("opp_team_elo", np.nan))
        elo_advantage = team_elo - opp_team_elo if not (np.isnan(team_elo) or np.isnan(opp_team_elo)) else np.nan
        channels.append(elo_advantage)

        vector = np.array(channels, dtype=np.float32)
        return self._handle_nans(vector)

    @staticmethod
    def _handle_nans(vector):
        vector = vector.copy()
        vector[np.isnan(vector)] = 0.0
        return vector

    def _normalize_target(self, target):
        if pd.isna(target):
            return -1

        if isinstance(target, (int, np.integer)):
            return int(target)

        if isinstance(target, float) and not np.isnan(target):
            return int(target)

        target_str = str(target).strip()
        return LABEL_MAP.get(target_str, -1)

    def _get_history(self, team_id):
        history = self.match_history.get(team_id, [])
        history_dates = self.match_history_dates.get(team_id, [])

        if len(history) < self.sequence_length:
            return None

        seq = np.array(history[-self.sequence_length:], dtype=np.float32)
        seq_dates = history_dates[-self.sequence_length:]
        last_date = seq_dates[-1] if len(seq_dates) > 0 else None

        return {
            "sequence": seq,
            "history_len": len(history),
            "history_dates": seq_dates,
            "last_date": last_date,
        }

    def process_matches(self, df):
        df = df.copy()

        required_cols = {"match_id", "role", "team", "target"}
        missing_required = required_cols - set(df.columns)
        if missing_required:
            raise ValueError(f"Missing required columns: {sorted(missing_required)}")

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        sort_cols = [c for c in ["Date", "match_id"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols).reset_index(drop=True)

        sequences_list = []
        sequences_cnn_list = []
        labels_list = []
        metadata_list = []

        for match_id, match_group in df.groupby("match_id", sort=False):
            if len(match_group) != 2:
                continue

            match_group = match_group.sort_values("role", ascending=False).reset_index(drop=True)
            home_row = match_group.iloc[0]
            away_row = match_group.iloc[1]

            home_team = home_row.get("team")
            away_team = away_row.get("team")
            target_date = self._safe_date(home_row.get("Date"))

            home_hist = self._get_history(home_team)
            away_hist = self._get_history(away_team)

            if home_hist is not None and away_hist is not None:
                home_hist_seq = home_hist["sequence"]
                away_hist_seq = away_hist["sequence"]

                seq = np.concatenate([home_hist_seq, away_hist_seq], axis=1)
                seq_cnn = np.transpose(seq, (1, 0))

                target = self._normalize_target(home_row.get("target", np.nan))

                sequences_list.append(seq)
                sequences_cnn_list.append(seq_cnn)
                labels_list.append(target)

                metadata_list.append({
                    "match_id": match_id,
                    "date": str(target_date) if target_date is not None else None,
                    "season": home_row.get("season"),
                    "home_team": home_team,
                    "away_team": away_team,
                    "target": target,
                    "sequence_length": self.sequence_length,
                    "home_history_len": home_hist["history_len"],
                    "away_history_len": away_hist["history_len"],
                    "home_history_last_date": str(home_hist["last_date"]) if home_hist["last_date"] is not None else None,
                    "away_history_last_date": str(away_hist["last_date"]) if away_hist["last_date"] is not None else None,
                    "home_history_end_before_target": (
                        home_hist["last_date"] is not None and target_date is not None and home_hist["last_date"] < target_date
                    ),
                    "away_history_end_before_target": (
                        away_hist["last_date"] is not None and target_date is not None and away_hist["last_date"] < target_date
                    ),
                })

            home_vector = self._build_feature_vector(home_row)
            away_vector = self._build_feature_vector(away_row)

            self.match_history[home_team].append(home_vector)
            self.match_history[away_team].append(away_vector)

            self.match_history_dates[home_team].append(target_date)
            self.match_history_dates[away_team].append(target_date)

        if len(sequences_list) == 0:
            raw_sequences = np.empty((0, self.sequence_length, len(self.combined_feature_names)), dtype=np.float32)
            cnn_sequences = np.empty((0, len(self.combined_feature_names), self.sequence_length), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int64)
            metadata = pd.DataFrame(columns=[
                "match_id",
                "date",
                "season",
                "home_team",
                "away_team",
                "target",
                "sequence_length",
                "home_history_len",
                "away_history_len",
                "home_history_last_date",
                "away_history_last_date",
                "home_history_end_before_target",
                "away_history_end_before_target",
            ])
        else:
            raw_sequences = np.array(sequences_list, dtype=np.float32)
            cnn_sequences = np.array(sequences_cnn_list, dtype=np.float32)
            labels = np.array(labels_list, dtype=np.int64)
            metadata = pd.DataFrame(metadata_list)

        return raw_sequences, cnn_sequences, labels, metadata


def save_feature_names(out_dir, builder):
    feature_payload = {
        "sequence_length": builder.sequence_length,
        "single_team_feature_names": builder.single_team_feature_names,
        "combined_feature_names": builder.combined_feature_names,
        "notes": {
            "raw_sequences_shape": "(num_samples, sequence_length, num_features)",
            "cnn_sequences_shape": "(num_samples, num_features, sequence_length)",
            "timestep_definition": "Each timestep is one past match before the target match.",
            "feature_definition": "Each timestep concatenates the home-team history vector and away-team history vector.",
        }
    }

    with open(os.path.join(out_dir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(feature_payload, f, indent=2)


def process_one_sequence_length(sequence_length):
    out_dir = os.path.join(OUTPUT_BASE_DIR, f"k_{sequence_length}")
    os.makedirs(out_dir, exist_ok=True)

    builder = CNNSequenceBuilder(sequence_length=sequence_length)
    save_feature_names(out_dir, builder)

    split_summaries = []

    for split_name in ["train", "val", "test"]:
        input_path = INPUT_FILES[split_name]

        if not os.path.exists(input_path):
            print(f"Skipping missing file: {input_path}")
            continue

        print(f"\nProcessing {split_name}...")
        df = pd.read_csv(input_path)
        print(f"  Input shape: {df.shape}")

        raw_sequences, cnn_sequences, labels, metadata = builder.process_matches(df)

        print(f"  Raw sequence shape: {raw_sequences.shape} (N, k, F)")
        print(f"  CNN shape:          {cnn_sequences.shape} (N, F, k)")
        print(f"  Labels shape:       {labels.shape}")

        valid_labels = labels[labels >= 0]
        if len(valid_labels) > 0:
            print(f"  Unique targets:     {np.unique(valid_labels)}")
        else:
            print("  Unique targets:     []")

        if len(metadata) > 0:
            print(
                "  Leak checks:        "
                f"home_ok={metadata['home_history_end_before_target'].all()}, "
                f"away_ok={metadata['away_history_end_before_target'].all()}"
            )

        npz_filename = f"k_{sequence_length}_{split_name}_sequences.npz"
        npz_path = os.path.join(SCRIPT_DIR, npz_filename)

        np.savez(
            npz_path,
            sequences=raw_sequences,
            sequences_cnn=cnn_sequences,
            labels=labels,
        )

        print(f"  Saved NPZ:          {npz_path}")

        metadata.to_csv(os.path.join(out_dir, f"{split_name}_metadata.csv"), index=False)

        split_summaries.append({
            "split": split_name,
            "num_samples": int(raw_sequences.shape[0]),
            "sequence_length": int(sequence_length),
            "num_features": int(raw_sequences.shape[2]) if raw_sequences.ndim == 3 and raw_sequences.shape[0] > 0 else len(builder.combined_feature_names),
            "raw_shape": list(raw_sequences.shape),
            "cnn_shape": list(cnn_sequences.shape),
            "labels_shape": list(labels.shape),
            "npz_file": npz_filename,
            "all_home_histories_before_target": bool(metadata["home_history_end_before_target"].all()) if len(metadata) > 0 else True,
            "all_away_histories_before_target": bool(metadata["away_history_end_before_target"].all()) if len(metadata) > 0 else True,
        })

    with open(os.path.join(out_dir, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(split_summaries, f, indent=2)

    print(f"\nSaved metadata/json outputs in: {out_dir}")
    print(f"Saved NPZ files in script directory: {SCRIPT_DIR}")


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    for sequence_length in SEQUENCE_LENGTHS:
        process_one_sequence_length(sequence_length)

    print("\nCNN sequence engineering complete.")
    print(f"Metadata and JSON files saved under: {OUTPUT_BASE_DIR}")
    print(f"NPZ files saved beside script in: {SCRIPT_DIR}")
    print("Saved versions for k = 3, 5, 10.")


if __name__ == "__main__":
    main()