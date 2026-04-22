"""Configuration loader — merges YAML config with environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"
_config: dict[str, Any] | None = None


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    global _config
    if _config is not None and path is None:
        return _config

    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if config_path.exists():
        with open(config_path) as f:
            _config = yaml.safe_load(f) or {}
    else:
        _config = {}

    # Override with env vars
    if api_key := os.getenv("OPENAI_API_KEY"):
        _config.setdefault("llm", {}).setdefault("openai", {})["api_key"] = api_key

    if provider := os.getenv("LLM_PROVIDER"):
        _config.setdefault("llm", {})["provider"] = provider

    if ollama_model := os.getenv("OLLAMA_MODEL"):
        _config.setdefault("llm", {}).setdefault("ollama", {})["model"] = ollama_model

    if ollama_host := os.getenv("OLLAMA_BASE_URL"):
        _config.setdefault("llm", {}).setdefault("ollama", {})["base_url"] = ollama_host.strip()

    # Optional overrides (used by web UI / automation without editing YAML)
    if (vision := os.getenv("VISION_BACKEND")) is not None:
        _config.setdefault("media_analysis", {})["vision_backend"] = vision.strip()
    if (wc := os.getenv("WRITER_CONSTRAINED")) is not None:
        _config.setdefault("writer", {})["constrained"] = wc.strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    if (mb := os.getenv("VIDEO_MOMENT_BACKEND")) is not None and mb.strip():
        _config.setdefault("media_analysis", {})["video_moment_backend"] = mb.strip().lower()

    if (ie := os.getenv("INTERNVIDEO2_ENABLED")) is not None:
        _config.setdefault("media_analysis", {}).setdefault("internvideo2", {})["enabled"] = (
            ie.strip().lower() in ("1", "true", "yes", "on")
        )

    if (im := os.getenv("INTERNVIDEO2_MODEL_ID")) is not None and im.strip():
        _config.setdefault("media_analysis", {}).setdefault("internvideo2", {})["model_id"] = im.strip()

    return _config


def get(key: str, default: Any = None) -> Any:
    """Dot-notation accessor, e.g. get('llm.provider')."""
    cfg = load_config()
    parts = key.split(".")
    current = cfg
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return default
        if current is None:
            return default
    return current
