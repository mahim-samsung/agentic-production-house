#!/usr/bin/env python3
"""Quick check that CLIP + Whisper stack loads (downloads weights on first run)."""
from __future__ import annotations

import sys
from pathlib import Path

# Optional: export HF_HOME="$(pwd)/.hf_cache" to keep Hugging Face weights inside the repo.
_root = Path(__file__).resolve().parent.parent


def main() -> int:
    sys.path.insert(0, str(_root))
    from src.utils.config import load_config

    load_config()
    import src.core.media as media

    print("Loading CLIP…")
    media._load_clip()
    if media._clip_model is None:
        print("CLIP failed (install: pip install -r requirements-ml.txt)")
        return 1
    print("CLIP OK")

    print("Importing Whisper…")
    import whisper  # noqa: F401

    print("Whisper OK (model downloads on first transcribe)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
