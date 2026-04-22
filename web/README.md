# Web UI

Next.js app: **/** — Produce (media, prompt, platform, output, per-run audio flags). **/admin** — technical defaults (Ollama model & base URL, vision backend, constrained writer, **moment backend** `heuristic` / `internvideo2` + InternVideo2 toggle & model id, optional server path to `config.yaml`). Admin values are stored in **localStorage** and applied to every run. The API forces **`LLM_PROVIDER=ollama`** for jobs from the web UI.

Runs `scripts/web_produce.py` in a background subprocess.

## Setup

From the **repository root**:

```bash
cd web
npm install
cp .env.local.example .env.local   # optional: set REPO_ROOT if needed
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Requirements

- Same stack as the CLI: **Python** + deps, **FFmpeg**, **Ollama** (local), optional ML / MusicGen.

## Paths

- Jobs: `<repo>/.tmp/web-jobs/<uuid>/` (input, `job.json`, `result.json`, copied `out/*.mp4`).
- Renders still go to `paths.output_dir` from config; the job folder holds a copy for download.
