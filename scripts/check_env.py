#!/usr/bin/env python3
"""Verify Ollama, FFmpeg, and Python deps before a full run."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import httpx
import yaml


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    cfg_path = root / "config.yaml"
    if not cfg_path.exists():
        print("config.yaml not found")
        return 1

    cfg = yaml.safe_load(cfg_path.read_text())
    ollama_cfg = cfg.get("llm", {}).get("ollama", {})
    base = ollama_cfg.get("base_url", "http://localhost:11434").rstrip("/")
    model = ollama_cfg.get("model", "llama3.2")

    ok = True

    if shutil.which("ffmpeg") is None:
        print("✗ ffmpeg not on PATH (required for video/audio)")
        ok = False
    else:
        print("✓ ffmpeg")

    try:
        r = httpx.get(f"{base}/api/tags", timeout=5.0)
        r.raise_for_status()
        names = {t.get("name", "") for t in r.json().get("models", [])}
        want = model.split(":")[0]
        has_model = any(n == model or n.split(":")[0] == want for n in names)
        if not has_model:
            print(f"✗ Ollama has no model matching '{model}'. Run: ollama pull {model}")
            ok = False
        else:
            print(f"✓ Ollama at {base}, model '{model}' available")
    except Exception as e:
        print(f"✗ Cannot reach Ollama at {base}: {e}")
        ok = False

    for mod in ("moviepy", "cv2", "PIL", "pydantic", "httpx", "yaml"):
        try:
            if mod == "PIL":
                __import__("PIL.Image")
            else:
                __import__(mod)
            print(f"✓ {mod}")
        except ImportError:
            print(f"✗ missing Python package for {mod} — run: pip install -r requirements.txt")
            ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
