#!/usr/bin/env python3
"""Quick check that vision + Whisper stack loads (downloads weights on first run)."""
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

    backend = str(media.cfg("media_analysis.vision_backend", "open_clip")).lower()
    if backend == "siglip2":
        print("Loading SigLIP2…")
        media._load_siglip2()
        if media._siglip2_model is None:
            print("SigLIP2 failed (install/verify: pip install -r requirements-ml.txt)")
            return 1
        print("SigLIP2 OK")
    else:
        print("Loading OpenCLIP…")
        media._load_clip()
        if media._clip_model is None:
            print("OpenCLIP failed (install: pip install -r requirements-ml.txt)")
            return 1
        print("OpenCLIP OK")

    print("Loading Whisper backend…")
    media._load_whisper()
    if media._whisper_model is None:
        print("Whisper failed (install/verify: pip install -r requirements-ml.txt)")
        return 1
    print(f"Whisper OK (backend={media._whisper_backend})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
