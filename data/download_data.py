import os
import csv
import requests
from io import StringIO

BASE = "https://raw.githubusercontent.com/datasets/football-datasets/main/datasets/premier-league"
OUT_DIR = os.path.join("data", "raw", "premier-league")

EXPECTED_COLUMNS = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "HTHG", "HTAG", "HTR", "Referee", "HS", "AS", "HST",
    "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"
]

START_SEASON = "0001"
END_SEASON = "2526"

def get_available_resources():
    url = f"{BASE}/datapackage.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    dp = r.json()

    files = []
    for resource in dp.get("resources", []):
        path = resource.get("path", "")
        if path.startswith("season-") and path.endswith(".csv"):
            files.append(path)

    return sorted(files)

def validate_csv(content, filename):
    text = content.decode("utf-8-sig")
    reader = csv.reader(StringIO(text))
    header = next(reader)

    missing = [c for c in EXPECTED_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"{filename} missing columns: {missing}")

def download_file(filename):
    url = f"{BASE}/{filename}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    validate_csv(r.content, filename)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, filename)

    with open(out_path, "wb") as f:
        f.write(r.content)

    print(f"Downloaded and validated: {filename}")

def season_code(filename):
    return filename.replace("season-", "").replace(".csv", "")

def main():
    available = get_available_resources()

    files = [
        f for f in available
        if START_SEASON <= season_code(f) <= END_SEASON
    ]

    for f in files:
        download_file(f)

    print(f"\nDownloaded {len(files)} validated files to {OUT_DIR}")

if __name__ == "__main__":
    main()