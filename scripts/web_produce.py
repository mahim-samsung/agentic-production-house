#!/usr/bin/env python3
"""
Run one production job for the Next.js web UI.

Usage:
    python scripts/web_produce.py /absolute/path/to/job_dir

Expects job_dir/job.json with keys:
  repo_root: absolute path to repository root
  prompt, platform, output_filename, skip_audio_enhance, generate_music
  config_path: optional absolute path to config.yaml (else repo default)

Media files must live under job_dir/input/
Optional music under job_dir/music/

Writes job_dir/result.json and copies the rendered video to job_dir/out/<output_filename>.
"""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path


def _fail(job_dir: Path, message: str, *, tb: str | None = None) -> None:
    payload = {"ok": False, "error": message}
    if tb:
        payload["traceback"] = tb
    (job_dir / "result.json").write_text(json.dumps(payload, indent=2))
    sys.exit(1)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: web_produce.py JOB_DIR", file=sys.stderr)
        sys.exit(2)

    job_dir = Path(sys.argv[1]).resolve()
    job_file = job_dir / "job.json"
    if not job_file.is_file():
        _fail(job_dir, "job.json missing")

    try:
        job = json.loads(job_file.read_text())
    except Exception as e:
        _fail(job_dir, f"Invalid job.json: {e}")

    repo_root = Path(job.get("repo_root", "")).resolve()
    if not repo_root.is_dir():
        _fail(job_dir, f"Invalid repo_root: {repo_root}")

    input_dir = job_dir / "input"
    if not input_dir.is_dir():
        _fail(job_dir, "input/ directory missing")

    media_files = sorted(
        p for p in input_dir.iterdir() if p.is_file() and not p.name.startswith(".")
    )
    if not media_files:
        _fail(job_dir, "No media files in input/")

    sys.path.insert(0, str(repo_root))
    import os

    os.chdir(repo_root)

    # Env overrides from job (applied on import via load_config)
    env_map = job.get("env") or {}
    if isinstance(env_map, dict):
        for k, v in env_map.items():
            if v is None:
                continue
            os.environ[str(k)] = str(v)

    from src.orchestrator import ProductionOrchestrator

    music_dir = job_dir / "music"
    music_arg = str(music_dir) if music_dir.is_dir() and any(music_dir.iterdir()) else None

    prompt = str(job.get("prompt", "")).strip()
    if not prompt:
        _fail(job_dir, "prompt is empty")

    platform = str(job.get("platform", "youtube_long"))
    output_filename = str(job.get("output_filename", "final_video.mp4"))
    skip_audio = bool(job.get("skip_audio_enhance", False))
    gen_music = bool(job.get("generate_music", False))
    config_path = job.get("config_path")
    config_arg = str(config_path) if config_path else None

    orch = ProductionOrchestrator(config_path=config_arg)
    try:
        report = orch.produce(
            prompt=prompt,
            media_files=[str(p) for p in media_files],
            output_filename=output_filename,
            platform=platform,
            music_dir=music_arg,
            skip_audio_enhance=skip_audio,
            generate_music=gen_music,
        )
    except Exception as e:
        _fail(job_dir, str(e), tb=traceback.format_exc())
    finally:
        try:
            orch.close()
        except Exception:
            pass

    out_dir = job_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    src_video = Path(report.output_path).resolve()
    dest_video = (out_dir / output_filename).resolve()
    try:
        shutil.copy2(src_video, dest_video)
    except Exception as e:
        _fail(job_dir, f"Video render succeeded but copy failed: {e}", tb=traceback.format_exc())

    report_src = src_video.parent / "production_report.json"
    report_dest = job_dir / "production_report.json"
    if report_src.is_file():
        try:
            shutil.copy2(report_src, report_dest)
        except Exception:
            pass

    result = {
        "ok": True,
        "output_path": str(dest_video),
        "report_path": str(report_dest) if report_dest.is_file() else str(report_src),
        "duration": report.duration,
        "resolution": report.resolution,
        "processing_time_seconds": report.processing_time_seconds,
        "title": report.creative_brief.title,
    }
    (job_dir / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        if len(sys.argv) >= 2:
            jd = Path(sys.argv[1]).resolve()
            if jd.is_dir():
                try:
                    _fail(jd, f"Worker crashed: {e}", tb=traceback.format_exc())
                except SystemExit:
                    raise
        else:
            traceback.print_exc()
            sys.exit(1)
