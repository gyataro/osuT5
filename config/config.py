from __future__ import annotations

import yaml

CONFIG_PATH = "./config/config.yaml"


class DotDict(dict):
    def __getattribute__(self, key: str):
        """Query dictionary with dot notation."""
        try:
            value = self[key]
        except KeyError:
            return super().__getattribute__(key)
        if isinstance(value, dict):
            return DotDict(value)
        return value


class Config(DotDict):
    def __init__(self, path: str = CONFIG_PATH):
        """Load project configurations from a .yaml file."""
        with open(path) as f:
            super().__init__(yaml.safe_load(f))
            f.close()
