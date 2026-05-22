from __future__ import annotations

import argparse

from src.config import load_config
from src.training.trainer import run_training
from src.utils import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a football match prediction model.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("project", "seed", default=42)))
    run_training(config)


if __name__ == "__main__":
    main()
