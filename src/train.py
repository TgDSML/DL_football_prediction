from __future__ import annotations

import argparse

from src.config import load_config
from src.pipelines.lstm_pipeline import LSTMPipeline
from src.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a football match prediction model.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--model",
        default="lstm",
        choices=["lstm", "transformer"],
        help="Model architecture to train.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("project", "seed", default=42)))

    if args.model == "lstm":
        pipeline = LSTMPipeline(config)
        pipeline.run()
    else:
        raise ValueError(f"Pipeline for {args.model} not yet implemented")


if __name__ == "__main__":
    main()
