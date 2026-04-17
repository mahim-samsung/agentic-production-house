#!/usr/bin/env bash
# One-time setup for Ollama + Python venv
# Usage: ./scripts/bootstrap.sh          # core deps only
#        ./scripts/bootstrap.sh --ml     # + PyTorch, OpenCLIP, Whisper (video understanding)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WITH_ML=false
if [[ "${1:-}" == "--ml" ]]; then
  WITH_ML=true
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "Install Ollama from https://ollama.com first."
  exit 1
fi

if ! ollama list >/dev/null 2>&1; then
  echo "Ollama is not running. Start the Ollama app, then run this script again."
  exit 1
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
if [[ "$WITH_ML" == true ]]; then
  echo "Installing ML stack (torch, open-clip-torch, whisper)…"
  pip install -r requirements-ml.txt
fi

MODEL="$(python3 - <<'PY'
import yaml
from pathlib import Path
c = yaml.safe_load(Path("config.yaml").read_text())
print(c["llm"]["ollama"]["model"])
PY
)"
echo "Pulling LLM: $MODEL (skip if already installed)"
ollama pull "$MODEL"

echo ""
echo "Setup done. Activate and run:"
echo "  source .venv/bin/activate"
if [[ "$WITH_ML" == true ]]; then
  echo "  python scripts/verify_ml.py   # optional: first CLIP download can take a few minutes"
fi
echo "  python main.py --prompt \"Your idea\" --media-dir ./input"
