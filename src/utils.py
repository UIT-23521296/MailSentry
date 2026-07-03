from __future__ import annotations

from pathlib import Path
import yaml


def load_config(path: str | Path = "configs/settings.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        # Fallback to project root (parent of src directory)
        config_path = Path(__file__).resolve().parent.parent / path
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
