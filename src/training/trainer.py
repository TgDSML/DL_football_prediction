from __future__ import annotations

from src.config import ExperimentConfig
from src.model import build_model
from src.utils import ensure_dir, get_device


def run_training(config: ExperimentConfig) -> None:
    """Create model and output directories.

    Replace the synthetic placeholder flow with real dataloaders once processed
    match sequences are available.
    """

    device = get_device()
    output_dirs = config.get("outputs", default={})
    for directory in output_dirs.values():
        ensure_dir(directory)

    model_type = str(config.get("training", "model_type", default="lstm"))
    model_kwargs = dict(config.get("model", default={}))
    model = build_model(model_type, **model_kwargs).to(device)

    print(f"Training pipeline initialized on {device}.")
    print(f"Model: {model.__class__.__name__}")
    print("Add processed data to enable full training.")
