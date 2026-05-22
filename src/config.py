from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """Thin wrapper around a nested YAML experiment config."""

    values: dict[str, Any]
    path: Path

    def get(self, *keys: str, default: Any = None) -> Any:
        current: Any = self.values
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        values = yaml.safe_load(file) or {}
    return ExperimentConfig(values=values, path=config_path)
