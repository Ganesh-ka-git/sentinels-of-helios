"""Configuration loading and path resolution."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config and resolve relative paths against the project root."""
    path = Path(path) if path else PROJECT_ROOT / "config" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, value in cfg.get("paths", {}).items():
        p = Path(value)
        cfg["paths"][key] = p if p.is_absolute() else PROJECT_ROOT / p
    return cfg


def require_email(cfg: dict[str, Any]) -> str:
    """Return the JSOC e-mail, failing loudly if the placeholder was not replaced."""
    email = cfg["jsoc"]["email"]
    if not email or email == "YOUR_JSOC_EMAIL" or "@" not in email:
        raise ValueError(
            "Set jsoc.email in config/config.yaml to the e-mail address registered at "
            "http://jsoc.stanford.edu/ajax/register_email.html"
        )
    return email
