from __future__ import annotations

import argparse

from src.config import load_config
from src.evaluation.metrics import summarize_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a football match prediction model.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    config = load_config(parse_args().config)
    print("Evaluation entry point is ready.")
    print(f"Results directory: {config.get('outputs', 'results_dir', default='outputs/results')}")
    print(summarize_metrics({}))


if __name__ == "__main__":
    main()
