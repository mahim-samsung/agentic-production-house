# Agentic Production House

Turn **your** clips and images plus a **prompt** into one edited video (with optional music) and a JSON **production report**.

## How it works

One pipeline: **Director** (brief from LLM) → **Analyst** (frames, scenes, speech, scores) → **Writer** (picks only **pre-validated** segments so trims stay real) → **Editor** (MoviePy/FFmpeg) → **Audio** (normalize, music / optional MusicGen).

**Example (travel footage)**

Put your clips in **`input/`** (airport, train, city walks, food, sunset — order does not matter; the edit plan orders them).

```bash
python main.py --prompt "One-day travel film: leave home, journey, explore the city, golden hour, quiet ending. Calm pacing, about 45 seconds." --media-dir ./input --platform vimeo_cinematic
```

Shorter social cut from the same folder:

```bash
python main.py --prompt "Travel hype: trains, streets, skyline, one beat per shot, high energy, ~25s" --media-dir ./input --platform tiktok
```

Use **`--media clip1.mp4 clip2.jpg …`** instead of **`--media-dir`** when you want an explicit file list. **`--platform`** picks presets in **`config.yaml`** → **`platforms`** (TikTok, YouTube, Reels, etc.).

---

## Run (CLI)

```bash
cd agentic-production-house
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip && pip install -r requirements.txt
```

**Ollama** (default LLM): install [Ollama](https://ollama.com), then pull the model in `config.yaml` (`llm.ollama.model`), e.g. `ollama pull llama3.3:70b`. Optional ML stack: `pip install -r requirements-ml.txt` and `python scripts/verify_ml.py`.

**FFmpeg** on `PATH` (e.g. `brew install ffmpeg`).

```bash
python main.py --prompt "Your idea here" --media-dir ./input
```

Common flags: `--platform …`, `--output name.mp4`, `--music-dir ./music`, `--generate-music`, `--no-audio`, `--config /path/to/config.yaml`. Check: `python scripts/check_env.py`.

---

## Run (Web UI)

Install front-end deps once (they live under **`web/`**):

```bash
cd web && npm install
```

Then either stay in **`web/`** and run **`npm run dev`**, or from the **repo root** run **`npm run dev`** (root `package.json` forwards into `web/`).

```bash
npm run dev
```

Open **http://localhost:3000** — **Produce** to run jobs; **Admin** for Ollama URL/model, vision, moment backend, `config.yaml` path (saved in the browser). Details: **`web/README.md`**.

---

## Configuration

Defaults live in **`config.yaml`** (LLM, `media_analysis`, `video`, `platforms`, `writer`, MusicGen, paths). Env overrides include `OLLAMA_MODEL`, `OLLAMA_BASE_URL`, `VISION_BACKEND`, `VIDEO_MOMENT_BACKEND`, `INTERNVIDEO2_*`, `WRITER_CONSTRAINED`, `LLM_PROVIDER`, `OPENAI_API_KEY`.

---

## License

Samsung Research Bangladesh. Third-party models (MusicGen, CLIP, Whisper weights, etc.) have their **own** terms.
