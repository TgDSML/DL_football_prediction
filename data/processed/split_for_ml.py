import os
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

INPUT_FILES = {
    "train": os.path.join(PROCESSED_DIR, "train_features.csv"),
    "val": os.path.join(PROCESSED_DIR, "val_features.csv"),
    "test": os.path.join(PROCESSED_DIR, "test_features.csv"),
}

OUTPUT_FULL_FILES = {
    "train": os.path.join(PROCESSED_DIR, "train_full_features.csv"),
    "val": os.path.join(PROCESSED_DIR, "val_full_features.csv"),
    "test": os.path.join(PROCESSED_DIR, "test_full_features.csv"),
}

TARGET_COL = "target"
DROP_COLS = [
    "match_id",
    "Date",
    "season",
    "split",
    "team",
    "opponent",
]


def main():
    for split_name in ["train", "val", "test"]:
        path = INPUT_FILES[split_name]
        df = pd.read_csv(path)

        if TARGET_COL not in df.columns:
            raise ValueError(f"{TARGET_COL} not found in {path}")

        df.to_csv(OUTPUT_FULL_FILES[split_name], index=False)

        y = df[[TARGET_COL]].copy()
        X = df.drop(columns=[c for c in DROP_COLS if c in df.columns] + [TARGET_COL], errors="ignore")

        X.to_csv(os.path.join(PROCESSED_DIR, f"X_{split_name}.csv"), index=False)
        y.to_csv(os.path.join(PROCESSED_DIR, f"y_{split_name}.csv"), index=False)

        print(f"Saved {split_name}: {len(df)} rows, {X.shape[1]} features")

    with open(os.path.join(PROCESSED_DIR, "feature_columns.txt"), "w", encoding="utf-8") as f:
        for col in X.columns:
            f.write(col + "\n")


if __name__ == "__main__":
    main()