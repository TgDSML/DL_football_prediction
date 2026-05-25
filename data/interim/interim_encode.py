import os
import json
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder

INPUT_DIR = os.path.join("data", "raw", "splits")
OUTPUT_DIR = os.path.join("data", "interim")
MAPPINGS_DIR = os.path.join(OUTPUT_DIR, "mappings")

CATEGORICAL_COLS = ["HomeTeam", "AwayTeam", "Referee", "season"]
RESULT_COLS = ["FTR", "HTR"]  # H=Home win, D=Draw, A=Away win

def clean_text_column(series):
    return (
        series.astype(str)
        .str.replace("\xa0", " ", regex=False)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

def normalize_referee_surname(name):
    name = str(name).replace("\xa0", " ").strip()
    name = re.sub(r"\s+", " ", name)
    if "," in name:
        surname = name.split(",")[0].strip()
    else:
        parts = name.split()
        surname = parts[-1] if parts else name
    surname = re.sub(r"[^\w\s'-]", "", surname)
    return surname

def canonicalize_referee(series):
    cleaned = clean_text_column(series)
    return cleaned.apply(normalize_referee_surname)

def add_date_features(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["HomeTeam"] = clean_text_column(df["HomeTeam"])
    df["AwayTeam"] = clean_text_column(df["AwayTeam"])
    df["season"] = clean_text_column(df["season"]).str.zfill(4)
    df["Referee"] = canonicalize_referee(df["Referee"])
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df

def fit_encoders_from_all_splits(train_df, val_df, test_df):
    encoders = {}
    combined = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(combined[col].astype(str))
        encoders[col] = le

    return encoders

def fit_result_encoders():
    """
    Fit fixed encoders for FTR and HTR.
    Always maps: A=0, D=1, H=2 (alphabetical, consistent across splits).
    """
    encoders = {}
    for col in RESULT_COLS:
        le = LabelEncoder()
        le.fit(["A", "D", "H"])  # fixed vocabulary: Away win, Draw, Home win
        encoders[col] = le
    return encoders

def transform_with_encoders(df, encoders):
    df = df.copy()
    for col, le in encoders.items():
        df[col] = le.transform(df[col].astype(str))
    return df

def save_mappings(encoders, label="categorical"):
    os.makedirs(MAPPINGS_DIR, exist_ok=True)
    for col, le in encoders.items():
        mapping = {cls: int(idx) for idx, cls in enumerate(le.classes_)}
        path = os.path.join(MAPPINGS_DIR, f"{col}_mapping.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        print(f"Saved {label} mapping: {path}")

def load_split(name):
    path = os.path.join(INPUT_DIR, f"{name}.csv")
    return pd.read_csv(path)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train = load_split("train")
    val   = load_split("val")
    test  = load_split("test")

    train = add_date_features(train)
    val   = add_date_features(val)
    test  = add_date_features(test)

    # --- Categorical encoders (team, referee, season) ---
    cat_encoders = fit_encoders_from_all_splits(train, val, test)
    train = transform_with_encoders(train, cat_encoders)
    val   = transform_with_encoders(val,   cat_encoders)
    test  = transform_with_encoders(test,  cat_encoders)
    save_mappings(cat_encoders, label="categorical")

    # --- Result encoders (FTR, HTR) ---
    result_encoders = fit_result_encoders()
    train = transform_with_encoders(train, result_encoders)
    val   = transform_with_encoders(val,   result_encoders)
    test  = transform_with_encoders(test,  result_encoders)
    save_mappings(result_encoders, label="result")

    train.to_csv(os.path.join(OUTPUT_DIR, "train_processed.csv"), index=False)
    val.to_csv(os.path.join(OUTPUT_DIR,   "val_processed.csv"),   index=False)
    test.to_csv(os.path.join(OUTPUT_DIR,  "test_processed.csv"),  index=False)

    print("\nSaved processed splits and mappings.")
    print(f"Train rows : {len(train)}")
    print(f"Val rows   : {len(val)}")
    print(f"Test rows  : {len(test)}")

    print("\n--- Categorical mappings ---")
    for col, le in cat_encoders.items():
        print(col, dict(zip(le.classes_, le.transform(le.classes_))))

    print("\n--- Result mappings ---")
    for col, le in result_encoders.items():
        print(col, dict(zip(le.classes_, le.transform(le.classes_))))

if __name__ == "__main__":
    main()