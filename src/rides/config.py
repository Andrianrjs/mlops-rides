"""Read one section of params.yaml (run commands from the repo root, like the rest of the pipeline)."""
from pathlib import Path

import yaml

PARAMS_FILE = Path("params.yaml")


def load_params(section: str, path: Path = PARAMS_FILE) -> dict:
    return yaml.safe_load(path.read_text())[section]
