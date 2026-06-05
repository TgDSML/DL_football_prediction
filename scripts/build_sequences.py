from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.sequences import (
    FEATURE_MODES,
    HOME_AWAY_PREFIX,
    HOME_ONLY_PREFIX,
    RAW_FEATURE_MODE,
    RAW_PLUS_ROLLING_FEATURE_MODE,
    SEQUENCE_LENGTH,
    SEQUENCE_OUTPUT_DIR,
    build_home_away_sequences,
    build_home_only_sequences,
    build_summary,
    save_home_only_sequences,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sequence datasets for scratch RNN baselines.")
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument(
        "--feature-mode",
        choices=FEATURE_MODES,
        default=RAW_FEATURE_MODE,
        help="Timestep feature representation to build.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["home_only", "home_away"],
        default=["home_only", "home_away"],
        help="Which sequence variants to build.",
    )
    return parser.parse_args()


def prefix_with_length(base_prefix: str, sequence_length: int) -> str:
    if sequence_length == SEQUENCE_LENGTH:
        return base_prefix
    return f"{base_prefix}_seq{sequence_length}"


def output_dir_for_feature_mode(feature_mode: str) -> Path:
    if feature_mode == RAW_FEATURE_MODE:
        return SEQUENCE_OUTPUT_DIR
    if feature_mode == RAW_PLUS_ROLLING_FEATURE_MODE:
        return SEQUENCE_OUTPUT_DIR / RAW_PLUS_ROLLING_FEATURE_MODE
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def main() -> None:
    args = parse_args()
    output_dir = output_dir_for_feature_mode(args.feature_mode)

    if "home_only" in args.variants:
        home_only_result = build_home_only_sequences(
            sequence_length=args.sequence_length,
            feature_mode=args.feature_mode,
        )
        home_only_paths = save_home_only_sequences(
            home_only_result,
            output_dir=output_dir,
            prefix=prefix_with_length(HOME_ONLY_PREFIX, args.sequence_length),
        )
        print(build_summary(home_only_result, home_only_paths, title="Home-Only Sequence Build Summary"))
        print()

    if "home_away" in args.variants:
        home_away_result = build_home_away_sequences(
            sequence_length=args.sequence_length,
            feature_mode=args.feature_mode,
        )
        home_away_paths = save_home_only_sequences(
            home_away_result,
            output_dir=output_dir,
            prefix=prefix_with_length(HOME_AWAY_PREFIX, args.sequence_length),
        )
        print(build_summary(home_away_result, home_away_paths, title="Home-Away Sequence Build Summary"))


if __name__ == "__main__":
    main()
