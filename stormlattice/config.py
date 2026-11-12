"""Load settings from config/settings.yaml and secrets from the environment (.env)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = REPO_ROOT / "config" / "settings.yaml"

load_dotenv(REPO_ROOT / ".env")


@lru_cache(maxsize=1)
def settings() -> dict[str, Any]:
    """Return the parsed settings.yaml (cached)."""
    with open(SETTINGS_PATH) as f:
        return yaml.safe_load(f)


def env(name: str, default: str | None = None) -> str | None:
    """Read a secret or deployment value from the environment."""
    return os.environ.get(name, default)


def data_dir() -> Path:
    """Local cache directory for downloads and outputs."""
    path = Path(env("STORM_DATA_DIR", str(REPO_ROOT / "data")))
    path.mkdir(parents=True, exist_ok=True)
    return path


def mode() -> str:
    """'live', 'replay' or 'simulated'. Shown as a banner in the interface."""
    value = (env("STORM_MODE", "replay") or "replay").lower()
    if value not in {"live", "replay", "simulated"}:
        raise ValueError(f"STORM_MODE must be live, replay or simulated, got {value!r}")
    return value
