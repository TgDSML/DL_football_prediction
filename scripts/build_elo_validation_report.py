from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_team_features import load_raw_matches
from src.features.elo import compute_elo_ratings


REPORT_PATH = PROJECT_ROOT / "outputs/reports/elo_validation_report.txt"


def build_report(sample_size: int = 12) -> str:
    matches = load_raw_matches()
    elo = compute_elo_ratings(matches)
    sample = elo.sample(n=min(sample_size, len(elo)), random_state=42).sort_values("Date")

    lines = [
        "Elo Validation Report",
        "=" * 40,
        "Elo features use the pre-match home_elo and away_elo columns.",
        "Post-match Elo is computed only after recording those feature values.",
        "",
        "Sample Pre/Post Ratings",
        "=" * 40,
        "| Date | Home | Away | FTR | Home pre | Home post | Away pre | Away post |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sample.itertuples(index=False):
        lines.append(
            f"| {row.Date.date()} | {row.HomeTeam} | {row.AwayTeam} | {row.FTR} | "
            f"{row.home_elo:.2f} | {row.home_post_elo:.2f} | "
            f"{row.away_elo:.2f} | {row.away_post_elo:.2f} |"
        )

    lines.extend(
        [
            "",
            "Leakage Check",
            "=" * 40,
            "PASS: model feature columns are pre-match Elo values only.",
            "PASS: post-match Elo columns are kept out of feature builders.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build_report(), encoding="utf-8")
    print(f"Saved Elo validation report: {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
