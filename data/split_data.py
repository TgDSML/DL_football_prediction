import os
import pandas as pd

INPUT_DIR = os.path.join("data", "raw", "premier-league")
OUTPUT_DIR = os.path.join("data", "raw", "splits")

TRAIN_START = "0001"
TRAIN_END = "1617"

VAL_START = "1718"
VAL_END = "2021"

TEST_START = "2122"   # everything after validation

def season_code(filename):
    return filename.replace("season-", "").replace(".csv", "")

def get_split(filename):
    code = season_code(filename)

    if TRAIN_START <= code <= TRAIN_END:
        return "train"
    elif VAL_START <= code <= VAL_END:
        return "val"
    elif code >= TEST_START:
        return "test"
    else:
        return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_dfs = []
    val_dfs = []
    test_dfs = []

    files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.startswith("season-") and f.endswith(".csv")
    )

    for f in files:
        split = get_split(f)
        if split is None:
            continue

        path = os.path.join(INPUT_DIR, f)
        df = pd.read_csv(path)
        df["season"] = season_code(f)

        if split == "train":
            train_dfs.append(df)
        elif split == "val":
            val_dfs.append(df)
        elif split == "test":
            test_dfs.append(df)

        print(f"Added {f} -> {split}")

    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    val_df = pd.concat(val_dfs, ignore_index=True) if val_dfs else pd.DataFrame()
    test_df = pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame()

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print(f"\nDone.")
    print(f"Train rows: {len(train_df)}")
    print(f"Val rows: {len(val_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Files saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
