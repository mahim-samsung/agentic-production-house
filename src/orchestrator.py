"""
Production Orchestrator — coordinates the full multi-agent pipeline.

Pipeline flow:
  1. Director   → CreativeBrief     (interprets user intent)
  2. Analyst    → MediaProfile[]    (understands available media)
  3. Writer     → EditDecisionList  (plans the edit)
  4. Editor     → video file        (assembles the video)
  5. Audio      → final video       (enhances audio)

Each step's output feeds into the next. The orchestrator also
produces a ProductionReport with full traceability.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from src.agents.analyst import AnalystAgent
from src.agents.audio import AudioAgent
from src.agents.director import DirectorAgent
from src.agents.editor import EditorAgent
from src.agents.writer import WriterAgent
from src.core.llm import LLMClient
from src.core.models import ProductionReport, TransitionType
from src.utils.config import get as cfg, load_config
from src.utils.logger import console, get_logger

log = get_logger("orchestrator")

_MUSIC_EXT = {".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a"}
_PLATFORM_PROMPT_HINTS = {
    "tiktok": "Target platform: TikTok short-form vertical video. Strong hook in first 1-2 seconds, punchy pacing, mobile-safe overlays.",
    "youtube_long": "Target platform: YouTube long-form. Prioritize coherent narrative flow, clarity, and sustained storytelling.",
    "youtube_shorts": "Target platform: YouTube Shorts. Vertical, quick opening hook, retention-focused pacing.",
    "youtube_highlights": "Target platform: YouTube highlights clip. Focus on strongest moments with clear context and payoff.",
    "vimeo_cinematic": "Target platform: Vimeo cinematic delivery. Emphasize visual quality, smoother transitions, and filmic pacing.",
    "instagram_reels": "Target platform: Instagram Reels. Vertical, expressive, polished pacing and caption-safe framing.",
    "instagram_stories": "Target platform: Instagram Stories. Vertical segmented storytelling with concise beats and clear visual hierarchy.",
}


def _music_library_nonempty(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(f.suffix.lower() in _MUSIC_EXT for f in path.iterdir() if f.is_file())


class ProductionOrchestrator:
    """Runs the full production pipeline from prompt to final video."""

    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            load_config(config_path)

        self.llm = LLMClient()

        self.director = DirectorAgent(self.llm)
        self.analyst = AnalystAgent(self.llm)
        self.writer = WriterAgent(self.llm)
        self.editor = EditorAgent(self.llm, output_dir=cfg("paths.output_dir", "output"))
        self.audio_agent = AudioAgent(self.llm)

    def produce(
        self,
        prompt: str,
        media_dir: Optional[str | Path] = None,
        media_files: Optional[list[str | Path]] = None,
        output_filename: str = "final_video.mp4",
        platform: str = "youtube_long",
        music_dir: Optional[str | Path] = None,
        skip_audio_enhance: bool = False,
        generate_music: bool = False,
    ) -> ProductionReport:
        """
        Run the full pipeline.

        Args:
            prompt: Natural-language description of the desired video.
            media_dir: Directory containing media files.
            media_files: Explicit list of media file paths.
            output_filename: Name for the output video.
            platform: Platform profile to optimize for (e.g. tiktok, youtube_shorts).
            music_dir: Directory with background music files.
            skip_audio_enhance: Skip the audio enhancement step.
            generate_music: If True (or music_generation.enabled in config), synthesize BGM with MusicGen.

        Returns:
            ProductionReport with the output path and full metadata.
        """
        start = time.time()
        console.rule("[bold cyan]AGENTIC PRODUCTION HOUSE[/]")
        console.print(f"[bold]Prompt:[/] {prompt}\n")

        platform_cfg = cfg(f"platforms.{platform}", {}) or {}
        profile_name = platform_cfg.get("name", platform)
        platform_prompt = _PLATFORM_PROMPT_HINTS.get(platform, "")
        guided_prompt = prompt if not platform_prompt else f"{prompt}\n\n{platform_prompt}"
        console.print(f"[bold]Platform:[/] {profile_name}")
        console.print()

        # ----- Step 1: Director -----
        console.rule("[bold magenta]Step 1 · Director[/]")
        brief = self.director.run(guided_prompt)
        if "target_duration" in platform_cfg:
            brief.target_duration = float(platform_cfg["target_duration"])
        if "transition_preference" in platform_cfg:
            brief.transition_preference = TransitionType(platform_cfg["transition_preference"])
        style_hint = platform_cfg.get("style_hint", "")
        if style_hint:
            existing = brief.style_notes.strip()
            brief.style_notes = f"{existing} | {style_hint}" if existing else style_hint
        console.print(f"  Title: [bold]{brief.title}[/]")
        console.print(f"  Mood: {brief.mood.value} | Pacing: {brief.pacing.value}")
        console.print(f"  Target: {brief.target_duration}s | Structure: {brief.structure}")
        console.print()

        # ----- Step 2: Analyst -----
        console.rule("[bold magenta]Step 2 · Analyst[/]")
        media_path_list = None
        if media_files:
            media_path_list = [Path(f) for f in media_files]

        profiles = self.analyst.run(
            media_dir=Path(media_dir) if media_dir else None,
            media_files=media_path_list,
            brief=brief,
        )
        console.print(f"  Analyzed {len(profiles)} assets")
        for p in profiles:
            console.print(
                f"    {Path(p.file_path).name}: "
                f"{p.media_type.value} | {p.duration:.1f}s | "
                f"relevance={p.relevance_score:.2f}"
            )
        console.print()

        # ----- Step 3: Writer -----
        console.rule("[bold magenta]Step 3 · Writer[/]")
        edl = self.writer.run(brief, profiles)
        console.print(f"  Segments: {len(edl.segments)} | Total: ~{edl.total_duration:.1f}s")
        for i, seg in enumerate(edl.segments):
            console.print(
                f"    {i+1}. {Path(seg.source_file).name} "
                f"[{seg.start_time:.1f}–{seg.end_time:.1f}s] "
                f"({seg.narrative_role})"
            )
        console.print()

        # ----- Step 4: Editor -----
        console.rule("[bold magenta]Step 4 · Editor[/]")
        self.editor.apply_render_profile(platform_cfg)
        video_path = self.editor.run(edl, output_filename)
        console.print(f"  Assembled: {video_path}")
        console.print()

        # ----- Step 5: Audio -----
        if not skip_audio_enhance:
            console.rule("[bold magenta]Step 5 · Audio[/]")
            music_path = Path(music_dir) if music_dir else Path(cfg("paths.assets_dir", "assets")) / "music"
            has_library = _music_library_nonempty(music_path)

            use_gen = bool(generate_music or cfg("music_generation.enabled", False))
            prefer_gen = bool(cfg("music_generation.prefer_over_library", True))

            generated_track: Optional[Path] = None
            if use_gen and (prefer_gen or not has_library):
                console.print("  [cyan]Synthesizing background music (MusicGen)…[/]")
                try:
                    from src.core.musicgen_client import generate_bgm_file

                    tmp_root = Path(cfg("paths.temp_dir", ".tmp"))
                    tmp_root.mkdir(parents=True, exist_ok=True)
                    out_wav = tmp_root / "generated_bgm.wav"
                    generated_track = generate_bgm_file(
                        brief=brief,
                        user_prompt=prompt,
                        video_duration=float(edl.total_duration or brief.target_duration),
                        output_path=out_wav,
                        edl=edl,
                    )
                    console.print(f"  [green]BGM saved:[/] {generated_track}")
                except Exception as e:
                    log.warning(f"MusicGen failed: {e}")
                    console.print(f"  [yellow]MusicGen failed ({e}); using library or no BGM.[/]")

            final_path = self.audio_agent.run(
                video_path=video_path,
                brief=brief,
                edl=edl,
                music_dir=music_path if music_path.exists() else None,
                music_file=generated_track if generated_track and generated_track.exists() else None,
            )
            # Replace original with enhanced version (atomic on same filesystem)
            if final_path.resolve() != video_path.resolve():
                video_path.unlink(missing_ok=True)
                final_path.rename(video_path)
                final_path = video_path
            console.print()
        else:
            final_path = video_path

        elapsed = time.time() - start

        # ----- Report -----
        console.rule("[bold green]Production Complete[/]")
        console.print(f"  Output: [bold]{final_path}[/]")
        console.print(f"  Duration: ~{edl.total_duration:.1f}s")
        console.print(f"  Processing time: {elapsed:.1f}s")

        report = ProductionReport(
            output_path=str(final_path),
            duration=edl.total_duration,
            resolution=(
                f"{platform_cfg.get('output_resolution', cfg('video.output_resolution', [1920,1080]))[0]}x"
                f"{platform_cfg.get('output_resolution', cfg('video.output_resolution', [1920,1080]))[1]}"
            ),
            creative_brief=brief,
            edit_decision_list=edl,
            media_profiles=profiles,
            processing_time_seconds=round(elapsed, 2),
        )

        report_path = Path(final_path).parent / "production_report.json"
        report_path.write_text(report.model_dump_json(indent=2))
        console.print(f"  Report: {report_path}")

        return report

    def close(self):
        self.llm.close()
